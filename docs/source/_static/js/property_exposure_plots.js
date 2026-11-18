(() => {
  const navy = "#001a64";
  const blue = "#1d4ed8";
  const mid = "#5b8def";
  const config = {responsive: true, displaylogo: false};
  const base = {
    paper_bgcolor: "white",
    plot_bgcolor: "white",
    font: {family: "Inter, Arial, sans-serif", size: 13},
    margin: {l: 60, r: 25, t: 70, b: 80},
  };

  function brand(layout) {
    layout.annotations = (layout.annotations || []).concat([
      {
        text: "PROTEUS",
        x: 1,
        y: -0.18,
        xref: "paper",
        yref: "paper",
        showarrow: false,
        xanchor: "right",
        font: {color: navy, size: 11},
      },
    ]);
    return layout;
  }

  function render() {
    const severityCenters = [
      1071428.5714285714, 3214285.714285714, 5357142.857142856,
      7500000.0, 9642857.142857142, 11785714.285714284,
      13928571.42857143, 16071428.57142857, 18214285.714285713,
      20357142.857142854, 22500000.0, 24642857.14285714,
      26785714.285714284, 28928571.428571425, 31071428.57142857,
      33214285.714285713, 35357142.85714285, 37500000.0,
      39642857.142857134, 41785714.28571428, 43928571.428571425,
      46071428.57142857, 48214285.71428572, 50357142.85714285,
      52500000.0, 54642857.142857134, 56785714.28571428,
      58928571.428571425, 61071428.57142857, 63214285.71428572,
      65357142.85714285, 67500000.0, 69642857.14285713,
      71785714.28571428, 73928571.42857143,
    ];
    const severityCounts = [
      13744, 3515, 2045, 1077, 931, 476, 379, 490, 212, 407, 141, 105,
      97, 86, 217, 63, 58, 57, 206, 22, 36, 25, 20, 160, 14, 11, 8, 7,
      11, 3, 7, 10, 7, 7, 192,
    ];
    const severityWidths = severityCenters.map(() => 2142857.142857143);

    Plotly.newPlot(
      "property-claim-severity",
      [{
        type: "bar",
        x: severityCenters,
        y: severityCounts,
        width: severityWidths,
        marker: {color: blue},
        hovertemplate: "Policy loss: %{x:,.0f}<br>Count: %{y:,}<extra></extra>",
      }],
      brand(Object.assign({}, base, {
        title: {text: "Portfolio Paid-Claim Severity"},
        xaxis: {title: {text: "Policy loss"}, tickformat: ",.0f"},
        yaxis: {title: {text: "Simulation count"}},
      })),
      config,
    );

    const layers = ["5m xs 5m", "10m xs 10m", "20m xs 20m"];
    Plotly.newPlot(
      "property-layer-rate",
      [
        {
          type: "bar",
          name: "Analytical exposure rate",
          x: layers,
          y: [0.1163531952, 0.1229171863, 0.1015502012],
          marker: {color: navy},
        },
        {
          type: "bar",
          name: "Simulated occurrence-only burn",
          x: layers,
          y: [0.1174244493, 0.1235477984, 0.1030516178],
          marker: {color: blue},
        },
        {
          type: "bar",
          name: "With £1m aggregate deductible",
          x: layers,
          y: [0.08962, 0.10715, 0.09551],
          marker: {color: mid},
        },
      ],
      brand(Object.assign({}, base, {
        title: {text: "Impact of Aggregate Terms on Layer Burn Rate"},
        barmode: "group",
        xaxis: {title: {text: "Layer"}},
        yaxis: {title: {text: "Rate on subject premium"}, tickformat: ".1%"},
        legend: {orientation: "h", y: -0.22},
      })),
      config,
    );
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", render);
  } else {
    render();
  }
})();
