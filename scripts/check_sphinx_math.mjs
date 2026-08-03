#!/usr/bin/env node
/** Validate every equation emitted by Sphinx with MathJax 4. */

import fs from "node:fs/promises";
import path from "node:path";

import { liteAdaptor } from "@mathjax/src/js/adaptors/liteAdaptor.js";
import { RegisterHTMLHandler } from "@mathjax/src/js/handlers/html.js";
import { TeX } from "@mathjax/src/js/input/tex.js";
import { mathjax } from "@mathjax/src/js/mathjax.js";
import { CHTML } from "@mathjax/src/js/output/chtml.js";
import "@mathjax/src/js/util/asyncLoad/esm.js";
import "@mathjax/src/js/input/tex/ams/AmsConfiguration.js";
import "@mathjax/src/js/input/tex/base/BaseConfiguration.js";
import "@mathjax/src/js/input/tex/newcommand/NewcommandConfiguration.js";
import "@mathjax/src/js/input/tex/textmacros/TextMacrosConfiguration.js";

const MATH_ELEMENT = /<(div|span)\b[^>]*class="math(?:\s[^"]*)?"[^>]*>([\s\S]*?)<\/\1>/g;
const MATHJAX_4_SCRIPT = /<script\b[^>]*\bsrc="[^"]*mathjax@4\/[^"]*"[^>]*>/i;
const MATHJAX_LINE_BREAKING = '"displayOverflow": "linebreak"';
const GENERATED_SOURCE_DIRECTORIES = new Set(["_modules", "_sources"]);
const LEAKED_MATH_MARKUP = [
  {
    pattern: /<p>\s*\.\.\s+math::/gi,
    problem: "unprocessed reStructuredText math directive",
  },
  {
    pattern: /:math:\s*<code\b/gi,
    problem: "unprocessed reStructuredText inline math role",
  },
  {
    pattern: /<p>\s*\$\$[\s\S]*?\$\$\s*<\/p>/gi,
    problem: "unprocessed dollar-delimited display math",
  },
];

function decodeHtml(value) {
  const namedEntities = new Map([
    ["amp", "&"],
    ["apos", "'"],
    ["gt", ">"],
    ["lt", "<"],
    ["nbsp", " "],
    ["quot", '"'],
  ]);
  return value.replace(/&(#x[0-9a-f]+|#\d+|[a-z]+);/gi, (entity, code) => {
    if (code.startsWith("#x")) {
      return String.fromCodePoint(Number.parseInt(code.slice(2), 16));
    }
    if (code.startsWith("#")) {
      return String.fromCodePoint(Number.parseInt(code.slice(1), 10));
    }
    return namedEntities.get(code.toLowerCase()) ?? entity;
  });
}

async function collectHtmlFiles(inputPath) {
  const stat = await fs.stat(inputPath);
  if (stat.isFile()) {
    return inputPath.endsWith(".html") ? [inputPath] : [];
  }

  const files = [];
  for (const entry of await fs.readdir(inputPath, { withFileTypes: true })) {
    const entryPath = path.join(inputPath, entry.name);
    if (entry.isDirectory() && !GENERATED_SOURCE_DIRECTORIES.has(entry.name)) {
      files.push(...(await collectHtmlFiles(entryPath)));
    } else if (entry.isFile() && entry.name.endsWith(".html")) {
      files.push(entryPath);
    }
  }
  return files;
}

function nearestAnchor(html, offset) {
  const prefix = html.slice(0, offset);
  const matches = [...prefix.matchAll(/\sid="([^"]+)"/g)];
  return matches.at(-1)?.[1] ?? "document";
}

function unwrapFormula(value) {
  const decoded = decodeHtml(value).trim();
  if (decoded.startsWith("\\[") && decoded.endsWith("\\]")) {
    return { display: true, tex: decoded.slice(2, -2).trim() };
  }
  if (decoded.startsWith("\\(") && decoded.endsWith("\\)")) {
    return { display: false, tex: decoded.slice(2, -2).trim() };
  }
  return null;
}

async function main() {
  const inputPath = process.argv[2];
  if (!inputPath) {
    console.error("usage: check_sphinx_math.mjs PATH_TO_HTML_OR_DIRECTORY");
    process.exitCode = 2;
    return;
  }

  const htmlFiles = await collectHtmlFiles(inputPath);
  const adaptor = liteAdaptor();
  RegisterHTMLHandler(adaptor);
  const tex = new TeX({ packages: ["base", "ams", "newcommand", "textmacros"] });
  const chtml = new CHTML({ fontURL: "" });
  const mathDocument = mathjax.document("", { InputJax: tex, OutputJax: chtml });

  let formulaCount = 0;
  let foundMathJax4 = false;
  let foundLineBreaking = false;
  const errors = [];
  for (const htmlFile of htmlFiles) {
    const html = await fs.readFile(htmlFile, "utf8");
    foundMathJax4 ||= MATHJAX_4_SCRIPT.test(html);
    foundLineBreaking ||= html.includes(MATHJAX_LINE_BREAKING);

    for (const check of LEAKED_MATH_MARKUP) {
      check.pattern.lastIndex = 0;
      for (const match of html.matchAll(check.pattern)) {
        const location = `${htmlFile}#${nearestAnchor(html, match.index)}`;
        errors.push(`${location}: ${check.problem}`);
      }
    }

    for (const match of html.matchAll(MATH_ELEMENT)) {
      formulaCount += 1;
      const location = `${htmlFile}#${nearestAnchor(html, match.index)}`;
      const formula = unwrapFormula(match[2]);
      if (formula === null) {
        errors.push(`${location}: math element has missing or unexpected delimiters`);
        continue;
      }
      if (!formula.tex) {
        errors.push(`${location}: empty math element`);
        continue;
      }

      try {
        const node = await mathDocument.convertPromise(formula.tex, { display: formula.display });
        const output = adaptor.outerHTML(node);
        if (output.includes("<mjx-merror")) {
          const preview = formula.tex.replace(/\s+/g, " ").slice(0, 160);
          errors.push(`${location}: MathJax could not render: ${preview}`);
        }
      } catch (error) {
        const preview = formula.tex.replace(/\s+/g, " ").slice(0, 160);
        errors.push(`${location}: ${String(error)}: ${preview}`);
      }
    }
  }

  if (!formulaCount) {
    errors.push(`${inputPath}: no rendered math elements found`);
  } else {
    if (!foundMathJax4) {
      errors.push(`${inputPath}: generated documentation does not load MathJax 4`);
    }
    if (!foundLineBreaking) {
      errors.push(`${inputPath}: generated documentation does not enable MathJax line breaking`);
    }
  }

  if (errors.length) {
    console.error("Invalid rendered mathematics:");
    for (const error of errors) {
      console.error(`- ${error}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log(`Validated ${formulaCount} rendered equations across ${htmlFiles.length} HTML files with MathJax.`);
}

await main();
