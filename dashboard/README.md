# Dashboard helpers

TypeScript for the React dashboard. No dependencies.

| File | What it does |
|---|---|
| `decodeFrame.ts` | Loads and decodes LGF1 frames (clip and live), the clip `manifest.json`, and gives hover values per cell (`cellInfo`) and the class table (`CLASSES`) |
| `drivability.ts` | Drivable / caution / blocked per cell from class, roughness and confidence, with on-road (default) and off-road profiles; drivable area in m² |

Format, units and rendering notes: [docs/FORMAT.md](../docs/FORMAT.md).
Data sources and the live API: [docs/DEPLOY.md](../docs/DEPLOY.md).

Drivability rules (on-road / off-road):

| Cells | On-road | Off-road |
|---|---|---|
| Vehicles, people, building, fence, trunk, pole, sign, unlabeled | blocked | blocked |
| Road, parking | drivable | drivable |
| Sidewalk, other-ground, terrain | caution | drivable |
| Vegetation | blocked | caution |
| Roughness ≥ 0.8 | blocked | blocked |
| Drivable but roughness ≥ 0.5 or confidence < 0.5 | caution | caution |

Areas without cells have no LiDAR data: show them as unknown, never as drivable.
