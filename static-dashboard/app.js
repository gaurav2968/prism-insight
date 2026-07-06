const DATA_URL = "./data/dashboard-data.json";

const DEFAULT_CONNECT_LINKS = [
  { label: "LinkedIn", url: "https://www.linkedin.com/in/kumar-gaurav-908942150/" },
  { label: "GitHub", url: "https://github.com/gaurav2968/" },
  { label: "Email", url: "mailto:rockstar.gs139@gmail.com" },
];

const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 2 });
const pct = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2, minimumFractionDigits: 2 });

function clsForNumber(value) {
  return value >= 0 ? "up" : "down";
}

function regimeBadgeClass(regime) {
  if (String(regime).includes("bull")) return "green";
  if (String(regime).includes("bear")) return "red";
  return "yellow";
}

function formatDecision(stock, day) {
  if (stock.was_entered) return { label: "Entered", className: "entered" };
  const maxPicks = Number(day.filters?.max_picks || day.max_picks || 0);
  if (maxPicks === 0) return { label: "No picks in this regime", className: "rejected" };
  return { label: `Not selected for top ${maxPicks || "N"}`, className: "rejected" };
}

function renderStats(summary) {
  const entries = [
    ["Open Positions", summary.open_positions],
    ["Closed Trades", summary.trade_count],
    ["Win Rate", `${summary.win_rate}%`],
    ["Avg Return", `${summary.avg_return >= 0 ? "+" : ""}${summary.avg_return}%`],
    ["Avg Hold", `${summary.avg_hold_days ?? 0}d`],
    ["Best Trade", `${summary.best_trade >= 0 ? "+" : ""}${summary.best_trade}%`],
  ];
  document.getElementById("stats").innerHTML = entries.map(([label, value]) => `
    <article class="stat">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${value}</div>
    </article>
  `).join("");
}

function renderBranding(profile = {}) {
  const brand = String(profile.brand_name || "AlphaPulse");
  const brandNode = document.getElementById("brandName");
  if (brandNode) {
    brandNode.textContent = brand;
  }
  document.title = `${brand} Dashboard`;
}

function renderSummaryPanels(summary, portfolio = {}) {
  const realizedPnlNet = Number(portfolio.realized_pnl_net ?? portfolio.realized_pnl_raw ?? summary.realized_pnl ?? 0);
  const equity = Number(portfolio.current_equity || 0);
  const equityRet = Number(portfolio.equity_return_pct || 0);
  document.getElementById("heroPnl").textContent = `${realizedPnlNet >= 0 ? "+" : ""}${money.format(realizedPnlNet)} net realized`;

  const capital = [
    ["Initial Capital", money.format(Number(portfolio.initial_capital || 0))],
    ["Current Equity", `${money.format(equity)} (${equityRet >= 0 ? "+" : ""}${pct.format(equityRet)}%)`],
    ["Open Holdings", `${portfolio.holdings_count ?? summary.open_positions ?? 0}`],
    ["Available Cash", money.format(Number(portfolio.available_cash || 0))],
    ["Open Exposure", money.format(Number(portfolio.open_exposure || 0))],
    ["Per Slot Capital", money.format(Number(portfolio.per_slot_capital || 0))],
    ["Wins / Losses", `${summary.win_count || 0} / ${summary.loss_count || 0}`],
    ["Last Exit", summary.latest_trade ? new Date(summary.latest_trade).toLocaleDateString() : "-"],
  ];
  document.getElementById("capitalSummary").innerHTML = capital.map(([label, value]) => `
    <div class="mini">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join("");

  const capitalNote = document.getElementById("capitalNote");
  if (capitalNote) {
    capitalNote.textContent = `Portfolio view uses a ${portfolio.max_slots || 5}-slot simulator model for capital and holdings summary.`;
  }

  const reasons = Object.entries(summary.exit_reason_counts || {});
  document.getElementById("exitReasons").innerHTML = reasons.length
    ? reasons.map(([reason, count]) => `
      <div class="reason-item">
        <span>${reason.replaceAll("_", " ")}</span>
        <strong>${count}</strong>
      </div>
    `).join("")
    : `<div class="muted">No historical exits in snapshot.</div>`;
}

function renderConnectLinks(profile = {}) {
  const root = document.getElementById("connectLinks");
  if (!root) return;
  const links = Array.isArray(profile.links) && profile.links.length ? profile.links : DEFAULT_CONNECT_LINKS;
  root.innerHTML = links.map((item) => {
    const label = item.label || "Platform";
    const url = item.url || "#";
    const safeUrl = String(url);
    return `
      <a class="connect-card" href="${safeUrl}" target="_blank" rel="noreferrer noopener">
        <span>${label}</span>
        <strong>${safeUrl}</strong>
      </a>
    `;
  }).join("");
}

function formatFilterValue(value, suffix = "") {
  if (value === null || value === undefined || value === "") {
    return "N/A";
  }
  return `${value}${suffix}`;
}

function titleizeRegime(regime) {
  return String(regime || "unknown").replaceAll("_", " ");
}

function renderRegimeGuide(triggerDays, marketContext = {}) {
  const root = document.getElementById("regimeGuide");
  if (!root) return;

  if (!triggerDays.length) {
    root.innerHTML = `<div class="muted">No regime data available in this snapshot.</div>`;
    return;
  }

  const latest = triggerDays[0];
  const regimeMeta = {
    bull_strong: "Above both MAs with strong slope and low volatility. Highest risk-on allocation.",
    bull_medium: "Above both MAs with one of slope/volatility favorable. Moderate participation.",
    bull_weak: "Above both MAs but weak internals. New entries are paused to avoid false breakouts.",
    slope_declining: "Price can be above MAs while trend quality deteriorates. Capital protection mode.",
    correction: "Pullback/transition phase. Capital protection is prioritized and entries are usually reduced or paused.",
    bear: "Risk-off regime. New positions are typically avoided and focus shifts to preserving capital.",
    bear_bottom: "Deep oversold regime. The algo may allow selective rebound candidates under tight filters.",
    bear_bottom_extreme: "Extreme oversold panic zone. Controlled rebound allocation with strict execution.",
    bull: "Legacy bull label. New runs classify into bull strong/medium/weak tiers.",
  };

  const regimeRules = [
    ["bull_strong", "Nifty above 50DMA and 200DMA, slope >= +1%, RVol <= 13%", "5 picks"],
    ["bull_medium", "Nifty above 50DMA and 200DMA, and (slope >= +1% OR RVol <= 13%)", "4 picks"],
    ["bull_weak", "Nifty above 50DMA and 200DMA, but slope < +1% and RVol > 13%", "0 picks"],
    ["slope_declining", "Nifty above both MAs but 50DMA slope is declining", "0 picks"],
    ["correction", "Nifty below either 50DMA or 200DMA", "0 picks"],
    ["bear", "Nifty below both 50DMA and 200DMA, RSI > 40", "0 picks"],
    ["bear_bottom", "Nifty below both MAs and RSI <= 40", "3 picks (quality >= 60)"],
    ["bear_bottom_extreme", "Nifty below both MAs and RSI <= 25", "3 picks (quality >= 40)"],
  ];

  const grouped = new Map();
  triggerDays.forEach((day) => {
    const key = String(day.regime_type || "unknown");
    if (!grouped.has(key)) {
      grouped.set(key, {
        regime: key,
        count: 0,
        lastSeen: day.date_display,
        filters: day.filters || {},
      });
    }
    grouped.get(key).count += 1;
  });

  const order = [
    "bull_strong",
    "bull_medium",
    "bull_weak",
    "slope_declining",
    "correction",
    "bear",
    "bear_bottom",
    "bear_bottom_extreme",
    "bull",
    "unknown",
  ];
  const rank = (regime) => {
    const index = order.indexOf(regime);
    return index === -1 ? order.length : index;
  };

  const currentFilters = latest.filters || {};
  const glossary = [
    ["Market Regime", "Classification of market conditions used to change risk appetite and selection strictness."],
    ["Nifty RSI", "Momentum/overbought-oversold signal for the index. Lower values generally imply weaker momentum."],
    ["Nifty Slope", "Directional slope of index trend. Positive implies upward drift, negative implies downward drift."],
    ["Min Quality", "Minimum quality score a stock must meet to qualify for selection under that regime."],
    ["Max Change", "Upper limit on recent price extension to avoid chasing over-stretched moves."],
    ["Max Picks", "Position cap for that day/regime. N/A means no new entries are allowed."],
    ["Quality Score", "Composite score from fundamentals and technical checks used in ranking candidates."],
    ["Risk/Reward", "Expected upside relative to downside from current price toward target and stop levels."],
    ["50DMA / 200DMA", "50-day and 200-day moving averages used as medium and long-term trend structure filters."],
    ["RVol", "Realized volatility from recent Nifty returns; lower values indicate cleaner trend conditions."],
  ];

  const maStructureMap = {
    above_both: "Above 50DMA & 200DMA",
    below_both: "Below 50DMA & 200DMA",
    mixed: "Mixed (between 50DMA and 200DMA)",
    unknown: "Not available",
  };

  const marketDate = marketContext.as_of ? new Date(marketContext.as_of).toLocaleDateString() : "-";
  const spot = marketContext.price != null ? money.format(Number(marketContext.price)) : "N/A";
  const dma50 = marketContext.sma50 != null ? money.format(Number(marketContext.sma50)) : "N/A";
  const dma200 = marketContext.sma200 != null ? money.format(Number(marketContext.sma200)) : "N/A";
  const slope50 = marketContext.sma50_slope_pct != null
    ? `${Number(marketContext.sma50_slope_pct) >= 0 ? "+" : ""}${pct.format(Number(marketContext.sma50_slope_pct))}%`
    : "N/A";
  const rvol = marketContext.rvol_pct != null ? `${pct.format(Number(marketContext.rvol_pct))}%` : "N/A";
  const structure = maStructureMap[String(marketContext.ma_structure || "unknown")] || maStructureMap.unknown;
  const vs50 = marketContext.position_vs_50dma || "N/A";
  const vs200 = marketContext.position_vs_200dma || "N/A";

  root.innerHTML = `
    <div class="regime-top">
      <article class="position-card regime-current">
        <div class="ticker-row">
          <div>
            <div class="ticker">Current Regime</div>
            <div class="sector">${latest.date_display}</div>
          </div>
          <span class="badge ${regimeBadgeClass(latest.regime_type)}">${titleizeRegime(latest.regime_type)}</span>
        </div>
        <p class="muted regime-note">${latest.message || regimeMeta[latest.regime_type] || "No regime note available."}</p>
        <div class="position-grid">
          <div class="mini"><span>Nifty Spot</span><strong>${spot}</strong></div>
          <div class="mini"><span>As Of</span><strong>${marketDate}</strong></div>
          <div class="mini"><span>Nifty 50DMA</span><strong>${dma50} (${vs50})</strong></div>
          <div class="mini"><span>Nifty 200DMA</span><strong>${dma200} (${vs200})</strong></div>
          <div class="mini"><span>50DMA Slope</span><strong>${slope50}</strong></div>
          <div class="mini"><span>RVol</span><strong>${rvol}</strong></div>
          <div class="mini"><span>MA Structure</span><strong>${structure}</strong></div>
          <div class="mini"><span>Nifty RSI</span><strong>${formatFilterValue(latest.nifty_rsi)}</strong></div>
          <div class="mini"><span>Nifty Slope</span><strong>${formatFilterValue(latest.nifty_slope, "%")}</strong></div>
          <div class="mini"><span>Min Quality</span><strong>${formatFilterValue(currentFilters.min_quality)}</strong></div>
          <div class="mini"><span>Max Picks</span><strong>${formatFilterValue(currentFilters.max_picks)}</strong></div>
        </div>
      </article>

      <article class="position-card regime-legend">
        <h3>How to Read This</h3>
        <div class="legend-list">
          ${Array.from(new Set([...regimeRules.map(([name]) => name), ...Array.from(grouped.keys())]))
            .sort((a, b) => rank(a) - rank(b))
            .map((name) => `
            <div class="legend-row">
              <span class="badge ${regimeBadgeClass(name)}">${titleizeRegime(name)}</span>
              <span>${regimeMeta[name] || "Custom regime state based on index behavior and filter policy."}</span>
            </div>
          `).join("")}
        </div>
      </article>
    </div>

    <div class="rulebook-grid">
      ${regimeRules.map(([name, condition, picks]) => `
        <article class="mini rule-card">
          <span>${titleizeRegime(name)}</span>
          <strong>${condition}</strong>
          <div class="rule-picks">${picks}</div>
        </article>
      `).join("")}
    </div>

    <div class="term-grid">
      ${glossary.map(([term, meaning]) => `
        <article class="mini term-card">
          <span>${term}</span>
          <strong>${meaning}</strong>
        </article>
      `).join("")}
    </div>
  `;
}

function renderOpenPositions(openPositions) {
  const root = document.getElementById("openPositions");
  if (!openPositions.length) {
    root.innerHTML = `<div class="muted">No open positions in the current snapshot.</div>`;
    return;
  }
  root.innerHTML = openPositions.map((position) => {
    const pnl = Number(position.return_pct || ((Number(position.current_price) - Number(position.entry_price)) / Number(position.entry_price)) * 100 || 0);
    return `
      <article class="position-card">
        <div class="ticker-row">
          <div>
            <div class="ticker">${position.ticker}</div>
            <div class="sector">${position.sector || ""}</div>
          </div>
          <div class="pnl ${clsForNumber(pnl)}">${pnl >= 0 ? "+" : ""}${pct.format(pnl)}%</div>
        </div>
        <div class="position-grid">
          <div class="mini"><span>Entry</span><strong>${money.format(Number(position.entry_price || 0))}</strong></div>
          <div class="mini"><span>Current</span><strong>${money.format(Number(position.current_price || 0))}</strong></div>
          <div class="mini"><span>Target</span><strong>${money.format(Number(position.take_profit || 0))}</strong></div>
          <div class="mini"><span>Stop</span><strong>${money.format(Number(position.stop_loss || 0))}</strong></div>
        </div>
      </article>
    `;
  }).join("");
}

function renderTrades(trades) {
  const body = document.querySelector("#tradeTable tbody");
  body.innerHTML = trades.map((trade) => {
    const exitReason = trade.scenario?.exit_reason || trade.exit_reason || "-";
    const profitRate = Number(trade.net_return_pct ?? trade.profit_rate ?? 0);
    const buyDate = trade.buy_date ? new Date(trade.buy_date).toLocaleDateString() : "-";
    const sellDate = trade.sell_date ? new Date(trade.sell_date).toLocaleDateString() : "-";
    return `
      <tr>
        <td>${trade.ticker}</td>
        <td>${buyDate}</td>
        <td>${sellDate}</td>
        <td>${money.format(Number(trade.buy_price || 0))}</td>
        <td>${money.format(Number(trade.sell_price || 0))}</td>
        <td>${Number(trade.quantity || 0)}</td>
        <td class="return ${clsForNumber(profitRate)}">${profitRate >= 0 ? "+" : ""}${pct.format(profitRate)}%</td>
        <td class="return ${clsForNumber(Number(trade.gross_pnl || 0))}">${Number(trade.gross_pnl || 0) >= 0 ? "+" : ""}${money.format(Number(trade.gross_pnl || 0))}</td>
        <td class="return ${clsForNumber(Number(trade.net_pnl || 0))}">${Number(trade.net_pnl || 0) >= 0 ? "+" : ""}${money.format(Number(trade.net_pnl || 0))}</td>
        <td>${money.format(Number(trade.total_charges || 0))}</td>
        <td>${exitReason}</td>
      </tr>
    `;
  }).join("");

  document.getElementById("recentTrades").innerHTML = trades.slice(0, 4).map((trade) => {
    const exitReason = trade.scenario?.exit_reason || trade.exit_reason || "-";
    const profitRate = Number(trade.net_return_pct ?? trade.profit_rate ?? 0);
    return `
      <article class="position-card">
        <div class="ticker-row">
          <div>
            <div class="ticker">${trade.ticker}</div>
            <div class="sector">${trade.trigger_type || trade.sector || "Trade"}</div>
          </div>
          <div class="pnl ${clsForNumber(profitRate)}">${profitRate >= 0 ? "+" : ""}${pct.format(profitRate)}%</div>
        </div>
        <div class="position-grid">
          <div class="mini"><span>Buy Date</span><strong>${trade.buy_date ? new Date(trade.buy_date).toLocaleDateString() : "-"}</strong></div>
          <div class="mini"><span>Sell Date</span><strong>${trade.sell_date ? new Date(trade.sell_date).toLocaleDateString() : "-"}</strong></div>
          <div class="mini"><span>Buy</span><strong>${money.format(Number(trade.buy_price || 0))}</strong></div>
          <div class="mini"><span>Sell</span><strong>${money.format(Number(trade.sell_price || 0))}</strong></div>
          <div class="mini"><span>Quantity</span><strong>${Number(trade.quantity || 0)}</strong></div>
          <div class="mini"><span>Held</span><strong>${trade.holding_days || 0} days</strong></div>
          <div class="mini"><span>Gross P&amp;L</span><strong class="${clsForNumber(Number(trade.gross_pnl || 0))}">${Number(trade.gross_pnl || 0) >= 0 ? "+" : ""}${money.format(Number(trade.gross_pnl || 0))}</strong></div>
          <div class="mini"><span>Net P&amp;L</span><strong class="${clsForNumber(Number(trade.net_pnl || 0))}">${Number(trade.net_pnl || 0) >= 0 ? "+" : ""}${money.format(Number(trade.net_pnl || 0))}</strong></div>
          <div class="mini"><span>Tax / Fees</span><strong>${money.format(Number(trade.total_charges || 0))}</strong></div>
          <div class="mini"><span>Exit</span><strong>${exitReason}</strong></div>
        </div>
      </article>
    `;
  }).join("");
}

function renderForwardChart(stock) {
  const series = stock.forward_series || [];
  if (!series.length) return "";
  const width = 300;
  const height = 150;
  const padding = { top: 12, right: 12, bottom: 30, left: 56 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = series.map((point) => Number(point.close));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = (max - min) || 1;
  const yMin = min - spread * 0.08;
  const yMax = max + spread * 0.08;
  const ySpread = (yMax - yMin) || 1;
  const firstDate = series[0]?.date || "";
  const midDate = series[Math.floor(series.length / 2)]?.date || firstDate;
  const lastDate = series[series.length - 1]?.date || "";
  const lastClose = values[values.length - 1] || 0;
  const entryPrice = Number(stock.current_price || 0);
  const targetPrice = Number(stock.target_price || 0);
  const stopPrice = Number(stock.stop_loss_price || 0);
  const xStep = series.length > 1 ? plotWidth / (series.length - 1) : plotWidth;
  const points = series.map((point, index) => {
    const x = padding.left + index * xStep;
    const y = padding.top + (1 - ((Number(point.close) - yMin) / ySpread)) * plotHeight;
    return `${x},${y}`;
  }).join(" ");
  const yTicks = [yMax, yMin + ySpread / 2, yMin].map((value) => Number(value.toFixed(2)));
  const dateLabel = (value) => new Date(value).toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const pointMeta = series.map((point, index) => {
    const x = padding.left + index * xStep;
    const y = padding.top + (1 - ((Number(point.close) - yMin) / ySpread)) * plotHeight;
    const deltaPct = entryPrice ? ((Number(point.close) - entryPrice) / entryPrice) * 100 : 0;
    return {
      x,
      y,
      date: point.date,
      close: Number(point.close),
      deltaPct,
    };
  });
  const guideLine = (price, className, label) => {
    if (!price) return "";
    const y = padding.top + (1 - ((price - yMin) / ySpread)) * plotHeight;
    return `
      <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="chart-guide ${className}"></line>
      <text x="${width - padding.right - 4}" y="${y - 4}" text-anchor="end" class="chart-guide-label ${className}">${label} ${money.format(price)}</text>
    `;
  };
  return `
    <div class="chart-wrap">
      <div class="chart-header">
        <div class="chart-label">30-day forward return path</div>
        <div class="chart-price">${money.format(lastClose)}</div>
      </div>
      <div class="chart-tooltip" hidden></div>
      <svg viewBox="0 0 ${width} ${height}" class="sparkline" preserveAspectRatio="none">
        ${yTicks.map((tick) => {
          const y = padding.top + (1 - ((tick - yMin) / ySpread)) * plotHeight;
          return `
            <line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" class="chart-grid"></line>
            <line x1="${padding.left - 4}" y1="${y}" x2="${padding.left}" y2="${y}" class="chart-axis-line"></line>
            <text x="${padding.left - 8}" y="${y + 4}" text-anchor="end" class="chart-tick-label">${money.format(tick)}</text>
          `;
        }).join("")}
        <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${padding.top + plotHeight}" class="chart-axis-line"></line>
        <line x1="${padding.left}" y1="${padding.top + plotHeight}" x2="${width - padding.right}" y2="${padding.top + plotHeight}" class="chart-axis-line"></line>
        ${guideLine(entryPrice, "entry", "Entry")}
        ${guideLine(targetPrice, "target", "TP")}
        ${guideLine(stopPrice, "stop", "SL")}
        <polyline points="${points}" fill="none" stroke="#78a8ff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"></polyline>
        <line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" y2="${padding.top + plotHeight}" class="chart-crosshair" visibility="hidden"></line>
        <circle cx="${padding.left}" cy="${padding.top}" r="4" class="chart-hover-dot" visibility="hidden"></circle>
        <circle cx="${padding.left}" cy="${padding.top + (1 - ((values[0] - yMin) / ySpread)) * plotHeight}" r="2.5" class="chart-point"></circle>
        <circle cx="${padding.left + (series.length - 1) * xStep}" cy="${padding.top + (1 - ((lastClose - yMin) / ySpread)) * plotHeight}" r="3" class="chart-point chart-point-end"></circle>
        ${pointMeta.map((point) => `
          <circle
            cx="${point.x}"
            cy="${point.y}"
            r="10"
            class="chart-hit-area"
            data-date="${point.date}"
            data-close="${point.close.toFixed(2)}"
            data-delta="${point.deltaPct.toFixed(2)}"
          ></circle>
        `).join("")}
        <text x="${padding.left}" y="${height - 10}" text-anchor="middle" class="chart-tick-label">${dateLabel(firstDate)}</text>
        <text x="${padding.left + plotWidth / 2}" y="${height - 10}" text-anchor="middle" class="chart-tick-label">${dateLabel(midDate)}</text>
        <text x="${width - padding.right}" y="${height - 10}" text-anchor="end" class="chart-tick-label">${dateLabel(lastDate)}</text>
      </svg>
    </div>
  `;
}

function renderWatchlist(triggerDays) {
  const root = document.getElementById("watchlist");
  root.innerHTML = triggerDays.map((day) => `
    <section class="watch-day">
      <div class="watch-head">
        <div>
          <h3>${day.date_display}</h3>
          <p class="muted">RSI ${day.nifty_rsi ?? "-"} · Slope ${day.nifty_slope ?? "-"}% · Max picks ${day.max_picks ?? 0}</p>
        </div>
        <div class="badge ${regimeBadgeClass(day.regime_type)}">${String(day.regime_type).replaceAll("_", " ")}</div>
      </div>
      ${day.message && !day.stocks.length ? `<p class="muted">${day.message}</p>` : ""}
      <div class="stocks-grid">
        ${day.stocks.map((stock) => {
          const decision = formatDecision(stock, day);
          return `
            <article class="stock-card">
              <div class="stock-top">
                <div>
                  <div class="ticker">${stock.ticker}</div>
                  <div class="sector">${stock.sector || stock.trigger_type}</div>
                </div>
                <div class="pnl ${clsForNumber(Number(stock.change_rate || 0))}">${Number(stock.change_rate || 0) >= 0 ? "+" : ""}${pct.format(Number(stock.change_rate || 0))}%</div>
              </div>
              <div class="stock-metrics">
                <div class="mini"><span>Current</span><strong>${money.format(Number(stock.current_price || 0))}</strong></div>
                <div class="mini"><span>Target</span><strong>${money.format(Number(stock.target_price || 0))}</strong></div>
                <div class="mini"><span>Stop</span><strong>${money.format(Number(stock.stop_loss_price || 0))}</strong></div>
                <div class="mini"><span>Quality</span><strong>${pct.format(Number(stock.quality_score || 0))}</strong></div>
              </div>
              ${renderForwardChart(stock)}
              <div class="decision ${decision.className}">${decision.label}</div>
            </article>
          `;
        }).join("")}
      </div>
    </section>
  `).join("");
}

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-button");
  const panels = document.querySelectorAll(".tab-panel");
  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const target = button.dataset.tabTarget;
      buttons.forEach((item) => item.classList.toggle("active", item === button));
      panels.forEach((panel) => panel.classList.toggle("active", panel.dataset.tabPanel === target));
    });
  });
}

function setupChartInteractions() {
  document.querySelectorAll(".chart-wrap").forEach((wrap) => {
    const tooltip = wrap.querySelector(".chart-tooltip");
    const crosshair = wrap.querySelector(".chart-crosshair");
    const hoverDot = wrap.querySelector(".chart-hover-dot");
    const hitAreas = wrap.querySelectorAll(".chart-hit-area");
    if (!tooltip || !crosshair || !hoverDot || !hitAreas.length) {
      return;
    }

    const hide = () => {
      tooltip.hidden = true;
      crosshair.setAttribute("visibility", "hidden");
      hoverDot.setAttribute("visibility", "hidden");
    };

    hitAreas.forEach((point) => {
      const show = (event) => {
        const date = point.dataset.date || "";
        const close = Number(point.dataset.close || 0);
        const delta = Number(point.dataset.delta || 0);
        tooltip.hidden = false;
        tooltip.innerHTML = `
          <strong>${new Date(date).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}</strong>
          <span>${money.format(close)}</span>
          <span class="${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "+" : ""}${pct.format(delta)}% vs entry</span>
        `;

        const rect = wrap.getBoundingClientRect();
  const x = event.clientX ?? rect.left + rect.width * 0.5;
  const y = event.clientY ?? rect.top + 24;
  const left = Math.min(Math.max(12, x - rect.left + 12), rect.width - 168);
  const top = Math.max(8, y - rect.top - 54);
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;

        const cx = point.getAttribute("cx");
        const cy = point.getAttribute("cy");
        crosshair.setAttribute("x1", cx);
        crosshair.setAttribute("x2", cx);
        crosshair.setAttribute("visibility", "visible");
        hoverDot.setAttribute("cx", cx);
        hoverDot.setAttribute("cy", cy);
        hoverDot.setAttribute("visibility", "visible");
      };

      point.addEventListener("pointerenter", show);
      point.addEventListener("pointermove", show);
      point.addEventListener("pointerleave", hide);
      point.addEventListener("pointerdown", show);
    });

    wrap.addEventListener("pointerleave", hide);
  });
}

async function boot() {
  const response = await fetch(DATA_URL, { cache: "no-store" });
  const payload = await response.json();
  renderBranding(payload.profile || {});
  document.getElementById("generatedAt").textContent = new Date(payload.generated_at).toLocaleString();
  renderStats(payload.summary || {});
  renderSummaryPanels(payload.summary || {}, payload.portfolio_view || {});
  renderRegimeGuide(payload.trigger_days || [], payload.market_context || {});
  renderConnectLinks(payload.profile || {});
  renderOpenPositions(payload.open_positions || []);
  renderTrades(payload.trades || []);
  renderWatchlist(payload.trigger_days || []);
  setupTabs();
  setupChartInteractions();
}

boot().catch((error) => {
  document.body.innerHTML = `<pre style="color:#fff;padding:24px;">Failed to load dashboard snapshot.\n${String(error)}</pre>`;
});