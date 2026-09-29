// drivability.ts: drivable / caution / blocked for each grid cell, computed in the dashboard.
//
// Uses only data already in every frame (class, terrain complexity, confidence), so
// profiles and thresholds can change live without re-running the pipeline.
// Areas with no cell have no LiDAR data: show them as "unknown", never as drivable.

import type { Frame } from './decodeFrame';

export type Drivability = 'drivable' | 'caution' | 'blocked';
export type VehicleProfile = 'on-road' | 'off-road';

export interface DrivabilityOptions {
  profile: VehicleProfile;
  roughCaution: number;   // terrain complexity (0-1) at or above which a drivable cell becomes caution
  roughBlocked: number;   // terrain complexity at or above which any cell is blocked
  minConfidence: number;  // semantic confidence below which a drivable cell becomes caution
}

export const DEFAULT_DRIVABILITY: DrivabilityOptions = {
  profile: 'on-road',
  roughCaution: 0.5,
  roughBlocked: 0.8,
  minConfidence: 0.5,
};

// Class ids (see CLASSES in decodeFrame.ts)
const OBSTACLES = new Set([1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 18, 19]); // vehicles, people, building, fence, trunk, pole, sign
const ROAD = new Set([9, 10]);                                          // road, parking
const OFF_ROAD_SURFACE = new Set([11, 12, 17]);                         // sidewalk, other-ground, terrain
const VEGETATION = 15;
const UNLABELED = 0;

export function cellDrivability(cls: number, complexity: number, confidence: number,
                                opts: DrivabilityOptions = DEFAULT_DRIVABILITY): Drivability {
  if (OBSTACLES.has(cls) || cls === UNLABELED) return 'blocked';
  if (complexity >= opts.roughBlocked) return 'blocked';

  let state: Drivability;
  if (ROAD.has(cls)) {
    state = 'drivable';
  } else if (OFF_ROAD_SURFACE.has(cls)) {
    state = opts.profile === 'off-road' ? 'drivable' : 'caution';
  } else if (cls === VEGETATION) {
    state = opts.profile === 'off-road' ? 'caution' : 'blocked';
  } else {
    state = 'caution';
  }
  if (state === 'drivable' && (complexity >= opts.roughCaution || confidence < opts.minConfidence)) {
    state = 'caution';
  }
  return state;
}

export const DRIVABILITY_CODE: Record<Drivability, number> = { drivable: 0, caution: 1, blocked: 2 };

/** One code per cell (0 drivable, 1 caution, 2 blocked), ready for an instanced-colour buffer. */
export function frameDrivability(frame: Frame, opts: DrivabilityOptions = DEFAULT_DRIVABILITY): Uint8Array {
  const c = frame.cells;
  const out = new Uint8Array(c.count);
  for (let i = 0; i < c.count; i++) {
    out[i] = DRIVABILITY_CODE[cellDrivability(c.cls[i], c.complexity[i] / 255, c.confidence[i] / 255, opts)];
  }
  return out;
}

/** Area (m²) in each state, e.g. for a stats card. Unknown area is not counted (no cells there). */
export function drivableArea(frame: Frame, codes: Uint8Array): Record<Drivability, number> {
  const area = { drivable: 0, caution: 0, blocked: 0 };
  const keys: Drivability[] = ['drivable', 'caution', 'blocked'];
  for (let i = 0; i < codes.length; i++) {
    const s = frame.header.level_sizes_m[frame.cells.level[i]];
    area[keys[codes[i]]] += s * s;
  }
  return area;
}

// Suggested colours; always pair them with the text label in the legend.
export const DRIVABILITY_COLORS: Record<Drivability | 'unknown', string> = {
  drivable: '#0ca30c',
  caution: '#fab219',
  blocked: '#d03b3b',
  unknown: '#6b7075',
};
