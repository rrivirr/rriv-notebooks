"""
Heat-pulse flow probe -- three-panel viewer.

One stacked panel per ring, each showing its eight thermistors A-H on a shared
time axis. Heater-on periods are shaded so pulses stay locatable.

Run:
    pip install dash plotly pandas numpy
    python app.py --csv flow_sensor_data_out.csv
    # http://127.0.0.1:8050

Fetch fresh data and run in one step:
    python app.py --device ac1f09fffe1a8575 --start 2026-07-30 --end 2026-08-06

Static HTML instead of a server:
    python app.py --csv flow_sensor_data_out.csv --html rings.html
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import flowvec as fv

# --------------------------------------------------------------------------
# Appearance
# --------------------------------------------------------------------------
BG = "#0e1621"
GRID = "#26364a"
INK = "#dbe6f0"
MUTE = "#7b8fa6"
HEAT = "#ff5c39"

#: One hue per sensor, held constant across all three rings so the same letter
#: is the same colour in every panel and can be compared ring to ring.
SENSOR_COLOR = {
    "A": "#4cc9f0",
    "B": "#4895ef",
    "C": "#7b6cf6",
    "D": "#b892ff",
    "E": "#f072c0",
    "F": "#ffa5a5",
    "G": "#ffd166",
    "H": "#8ce99a",
}


def fig_rings(df, cycles, mode: str = "raw") -> go.Figure:
    """Three stacked panels, one per ring, eight sensor traces each.

    mode "raw"    plots temperature as recorded.
    mode "spread" plots each sensor minus its own ring's mean, which removes the
                  common drift the whole probe shares and leaves only the
                  between-sensor differences the vector method actually uses.
    """
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.055,
        subplot_titles=[f"Ring {r}" for r in fv.RINGS],
    )

    # Plotly writes datetime axes as ISO strings, one full copy per trace, which
    # on a multi-day 24-sensor record turns a static export into tens of MB.
    # Epoch milliseconds plus an explicit date axis is byte-identical to read but
    # base64-encodes instead, and float32 is far below the sensor's 0.01 °C step.
    tstamp = df["timestamp"].to_numpy().astype("datetime64[ms]").astype("int64")

    for i, r in enumerate(fv.RINGS, start=1):
        cols = [f"Ring{r}_Sensor{s}" for s in fv.SENSORS]
        ring_mean = df[cols].mean(axis=1)
        for s in fv.SENSORS:
            col = f"Ring{r}_Sensor{s}"
            y = df[col] - ring_mean if mode == "spread" else df[col]
            fig.add_trace(
                go.Scattergl(
                    x=tstamp,
                    y=y.to_numpy().astype("float32"),
                    name=s,
                    legendgroup=s,
                    showlegend=(i == 1),
                    line=dict(color=SENSOR_COLOR[s], width=1),
                    hovertemplate=f"R{r}{s}  %{{y:.3f}} °C<extra></extra>",
                ),
                row=i,
                col=1,
            )

        for _, c in cycles.iterrows():
            fig.add_vrect(
                x0=c["on_start"],
                x1=c["on_end"],
                fillcolor=HEAT,
                opacity=0.22,
                line_width=0,
                layer="below",
                row=i,
                col=1,
            )

    ytitle = "ΔT from ring mean (°C)" if mode == "spread" else "°C"
    fig.update_layout(
        height=820,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="ui-monospace, SFMono-Regular, Menlo, monospace", size=12),
        margin=dict(l=62, r=24, t=34, b=44),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            y=1.045,
            x=1,
            xanchor="right",
            bgcolor="rgba(0,0,0,0)",
            title_text="",
        ),
    )
    fig.update_xaxes(type="date", gridcolor=GRID, zerolinecolor=GRID, showspikes=True,
                     spikecolor=MUTE, spikethickness=1, spikemode="across")
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, title_text=ytitle)
    for ann in fig.layout.annotations:
        ann.font.update(size=11, color=MUTE)
        ann.update(x=0, xanchor="left")
    return fig


# --------------------------------------------------------------------------
# Data fetch
# --------------------------------------------------------------------------

def fetch(device: str, start: str, end: str, out: str) -> str:
    url = (
        f"https://api.rriv.org/data/readings/{device}"
        f"?rangeStart={start}&rangeEnd={end}&format=csv"
    )
    print(f"fetching {url}")
    with open(out, "wb") as fh:
        subprocess.run(["curl", "-sS", url], stdout=fh, check=True)
    return out


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def build_app(csv_path: str):
    from dash import Dash, dcc, html, Input, Output

    df = fv.load(csv_path)
    cycles = fv.find_cycles(df)

    app = Dash(__name__)
    app.title = "Heat-pulse flow probe"
    app.layout = html.Div(
        style={
            "background": BG,
            "minHeight": "100vh",
            "padding": "22px 26px",
            "fontFamily": "ui-monospace, SFMono-Regular, Menlo, monospace",
            "color": INK,
        },
        children=[
            dcc.RadioItems(
                id="mode",
                options=[
                    {"label": "  Temperature", "value": "raw"},
                    {"label": "  Spread from ring mean", "value": "spread"},
                ],
                value="raw",
                inline=True,
                style={"fontSize": "12px", "marginBottom": "6px"},
                inputStyle={"marginRight": "4px", "marginLeft": "16px"},
            ),
            dcc.Graph(id="rings", config={"displaylogo": False}),
        ],
    )

    @app.callback(Output("rings", "figure"), Input("mode", "value"))
    def _update(mode):
        return fig_rings(df, cycles, mode)

    return app


def write_html(csv_path: str, out_path: str, mode: str = "raw") -> str:
    df = fv.load(csv_path)
    cycles = fv.find_cycles(df)
    fig = fig_rings(df, cycles, mode)
    body = pio.to_html(fig, include_plotlyjs="cdn", full_html=False,
                       config={"displaylogo": False})
    with open(out_path, "w") as fh:
        fh.write(
            f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>Heat-pulse flow probe</title>"
            f"<style>:root{{color-scheme:dark}}body{{background:{BG};margin:0;padding:18px}}</style>"
            f"</head><body>{body}</body></html>"
        )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="flow_sensor_data_out.csv")
    p.add_argument("--device", help="RRIV device id; fetches fresh data before starting")
    p.add_argument("--start", default="2026-07-30")
    p.add_argument("--end", default="2026-08-06")
    p.add_argument("--html", help="write a standalone HTML file instead of serving")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()

    csv_path = fetch(args.device, args.start, args.end, args.csv) if args.device else args.csv

    try:
        if args.html:
            print("wrote", write_html(csv_path, args.html))
            return
        build_app(csv_path).run(debug=False, port=args.port)
    except FileNotFoundError:
        sys.exit(f"no such file: {csv_path} — pass --csv or --device")


if __name__ == "__main__":
    main()
