# Heat-pulse flow probe — three-panel viewer

One panel per ring (1 = top, 2 = middle, 3 = bottom), each showing its eight
thermistors A–H on a shared time axis. Heater-on periods are shaded.

## Run

```bash
pip install dash plotly pandas numpy

python app.py --csv flow_sensor_data_out.csv          # serve at 127.0.0.1:8050
python app.py --device ac1f09fffe1a8575 --start 2026-07-30 --end 2026-08-06
python app.py --csv flow_sensor_data_out.csv --html rings.html   # static file
```

## Panels

Sensor A–H keeps the same colour in all three panels, so a letter can be compared
ring to ring. Clicking a letter in the legend toggles it in all three at once.
The x-axes are linked: zoom one panel and the others follow.

One toggle at the top:

- **Temperature** — as recorded.
- **Spread from ring mean** — each sensor minus its own ring's mean. Removes the
  common drift the whole probe shares and leaves only the between-sensor
  differences a heat pulse would create.

## Files

| file | what it does |
|---|---|
| `app.py` | The viewer. |
| `flowvec.py` | Loading, cleaning, and heater-cycle detection. Column map and geometry live at the top. |

`flowvec.py` still carries the vector-decomposition functions (`analyse_all`,
`stack_cycles`) if you want them from a notebook, but nothing in the viewer calls
them.

## Data quirks handled on load

- **Misaligned rows** — the API left-pads short records with zeros, shifting every
  reading right so `value_25` holds a temperature instead of the heater flag.
  Flagged and blanked; unrecoverable without knowing how many values dropped.
- **Dropouts** — sensors reporting exactly `0.0`, a failed read rather than a real
  0 °C measurement. Converted to NaN so they leave a gap in the trace instead of a
  spike to zero.

In the 2026-07-31 → 08-04 export that is 1,962 and 3,471 rows respectively.
