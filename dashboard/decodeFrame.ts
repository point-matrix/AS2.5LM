// decodeFrame.ts: read LGF1 LiDAR frames (points + 2.5D grid cells) in the browser.
//
// Frames come from two places, in the same format:
//   - the precomputed clip on the Hugging Face dataset (manifest.json + lite/ and full/ frames)
//   - the live Space, for a single uploaded KITTI .bin
// Spec: FORMAT.md. No dependencies; arrays are views on the downloaded buffer (no copying).

export type SectionDtype = 'int16' | 'uint8' | 'uint16' | 'float32';

export interface SectionInfo {
  name: string;
  dtype: SectionDtype;
  shape: number[];
  offset: number;
  bytes: number;
}

export interface FrameHeader {
  format: 'LGF1';
  version: number;
  frame_id: string;
  source: 'precomputed' | 'live';
  sequence: string | null;
  n_points_total: number;   // points in the scan before thinning
  point_stride: number;     // 1 = every point stored, 4 = every 4th point
  n_cells: number;
  level_sizes_m: number[];  // cell size by level: [0.4, 0.2, 0.1, 0.05]
  sensor_height_m: number;  // add to z to put the road near 0 m
  stats: Record<string, number>;
  timing_ms: Record<string, number>;
  device: string | null;
  accuracy: { accuracy: number; car_iou: number | null; person_iou: number | null } | null;
  sections: SectionInfo[];
  [key: string]: unknown;
}

export interface FramePoints {
  count: number;
  xyzCm: Int16Array;          // x0, y0, z0, x1, ... in centimetres, sensor frame
  intensity: Uint8Array;      // 0-255 = remission 0-1
  cls: Uint8Array;            // predicted class id (see CLASSES)
  confidence: Uint8Array;     // 0-255 = 0-1
  gtClass: Uint8Array | null; // ground-truth class id (precomputed frames only)
}

export interface FrameCells {
  count: number;
  ixy: Int16Array;            // ix0, iy0, ix1, ... cell index at its own level
  level: Uint8Array;          // 3 = 5 cm, 2 = 10 cm, 1 = 20 cm, 0 = 40 cm
  cls: Uint8Array;            // semantic class id
  zCm: Int16Array;            // min0, max0, mean0, min1, ... in centimetres
  zStdMm: Uint16Array;        // height standard deviation, millimetres
  traversability: Uint8Array; // 0-255 = 0-1
  complexity: Uint8Array;     // terrain complexity, 0-255 = 0-1
  confidence: Uint8Array;     // semantic confidence, 0-255 = 0-1
  pointCount: Uint16Array;
}

export interface Frame {
  header: FrameHeader;
  points: FramePoints;
  cells: FrameCells;
}

export interface CellInfo {
  index: number;
  cls: number;
  className: string;
  level: number;
  size: number;               // metres
  xMin: number; xMax: number; yMin: number; yMax: number;
  zMin: number; zMax: number; zMean: number; zStd: number;  // metres
  traversability: number;     // 0-1
  complexity: number;         // 0-1
  confidence: number;         // 0-1
  pointCount: number;
}

export interface ClassInfo {
  id: number;
  name: string;
  color: string;
  priority: number;
  traversability: number;
}

// Same table as grid_engine_v2/config.py (SalsaNext class ids).
export const CLASSES: ClassInfo[] = [
  { id: 0, name: 'unlabeled', color: '#000000', priority: 5, traversability: 0.0 },
  { id: 1, name: 'car', color: '#ff0000', priority: 5, traversability: 0.1 },
  { id: 2, name: 'bicycle', color: '#ff8000', priority: 5, traversability: 0.1 },
  { id: 3, name: 'motorcycle', color: '#ff8000', priority: 5, traversability: 0.1 },
  { id: 4, name: 'truck', color: '#800000', priority: 5, traversability: 0.1 },
  { id: 5, name: 'other-vehicle', color: '#800000', priority: 5, traversability: 0.1 },
  { id: 6, name: 'person', color: '#0000ff', priority: 5, traversability: 0.05 },
  { id: 7, name: 'bicyclist', color: '#0000ff', priority: 5, traversability: 0.05 },
  { id: 8, name: 'motorcyclist', color: '#0000ff', priority: 5, traversability: 0.05 },
  { id: 9, name: 'road', color: '#808080', priority: 1, traversability: 0.95 },
  { id: 10, name: 'parking', color: '#606060', priority: 1, traversability: 0.9 },
  { id: 11, name: 'sidewalk', color: '#a0a0a0', priority: 1, traversability: 0.85 },
  { id: 12, name: 'other-ground', color: '#c0c0c0', priority: 1, traversability: 0.7 },
  { id: 13, name: 'building', color: '#ffd700', priority: 4, traversability: 0.05 },
  { id: 14, name: 'fence', color: '#8b4513', priority: 4, traversability: 0.05 },
  { id: 15, name: 'vegetation', color: '#008000', priority: 2, traversability: 0.4 },
  { id: 16, name: 'trunk', color: '#8b4513', priority: 4, traversability: 0.05 },
  { id: 17, name: 'terrain', color: '#90ee90', priority: 1, traversability: 0.5 },
  { id: 18, name: 'pole', color: '#d3d3d3', priority: 4, traversability: 0.05 },
  { id: 19, name: 'traffic-sign', color: '#ff00ff', priority: 4, traversability: 0.05 },
];

const MAGIC = 'LGF1';

function isGzip(buffer: ArrayBuffer): boolean {
  const b = new Uint8Array(buffer, 0, Math.min(2, buffer.byteLength));
  return b.length === 2 && b[0] === 0x1f && b[1] === 0x8b;
}

/** Decompress a .gz buffer with the browser's built-in DecompressionStream. */
export async function gunzipIfNeeded(buffer: ArrayBuffer): Promise<ArrayBuffer> {
  if (!isGzip(buffer)) return buffer;
  const stream = new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'));
  return await new Response(stream).arrayBuffer();
}

function view(buffer: ArrayBuffer, s: SectionInfo): Int16Array | Uint8Array | Uint16Array | Float32Array {
  const count = s.shape.reduce((a, b) => a * b, 1);
  switch (s.dtype) {
    case 'int16': return new Int16Array(buffer, s.offset, count);
    case 'uint8': return new Uint8Array(buffer, s.offset, count);
    case 'uint16': return new Uint16Array(buffer, s.offset, count);
    case 'float32': return new Float32Array(buffer, s.offset, count);
    default: throw new Error(`LGF1: unsupported dtype ${String(s.dtype)} in section ${s.name}`);
  }
}

/** Decode an uncompressed LGF1 buffer. Use loadFrame() for URLs and .gz files. */
export function decodeFrame(buffer: ArrayBuffer): Frame {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(bytes[0], bytes[1], bytes[2], bytes[3]);
  if (magic !== MAGIC) {
    throw new Error(isGzip(buffer) ? 'LGF1: buffer is gzip-compressed, call gunzipIfNeeded() first'
                                   : 'LGF1: not a frame file (bad magic)');
  }
  const headerLen = new DataView(buffer).getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerLen))) as FrameHeader;
  const s: Record<string, SectionInfo> = {};
  for (const sec of header.sections) s[sec.name] = sec;
  const get = <T>(name: string): T => {
    if (!s[name]) throw new Error(`LGF1: missing section ${name}`);
    return view(buffer, s[name]) as unknown as T;
  };
  const nPoints = s['points.class'].shape[0];
  const nCells = s['cells.class'].shape[0];
  return {
    header,
    points: {
      count: nPoints,
      xyzCm: get<Int16Array>('points.xyz_cm'),
      intensity: get<Uint8Array>('points.intensity'),
      cls: get<Uint8Array>('points.class'),
      confidence: get<Uint8Array>('points.confidence'),
      gtClass: s['points.gt_class'] ? get<Uint8Array>('points.gt_class') : null,
    },
    cells: {
      count: nCells,
      ixy: get<Int16Array>('cells.ixy'),
      level: get<Uint8Array>('cells.level'),
      cls: get<Uint8Array>('cells.class'),
      zCm: get<Int16Array>('cells.z_cm'),
      zStdMm: get<Uint16Array>('cells.z_std_mm'),
      traversability: get<Uint8Array>('cells.traversability'),
      complexity: get<Uint8Array>('cells.complexity'),
      confidence: get<Uint8Array>('cells.confidence'),
      pointCount: get<Uint16Array>('cells.point_count'),
    },
  };
}

/** Fetch, decompress if needed, and decode a frame. */
export async function loadFrame(url: string, init?: RequestInit): Promise<Frame> {
  const res = await fetch(url, init);
  if (!res.ok) throw new Error(`LGF1: ${res.status} ${res.statusText} for ${url}`);
  return decodeFrame(await gunzipIfNeeded(await res.arrayBuffer()));
}

/** Everything about cell i in metres / 0-1 units (for hover panels). */
export function cellInfo(frame: Frame, i: number): CellInfo {
  const c = frame.cells;
  const level = c.level[i];
  const size = frame.header.level_sizes_m[level];
  const ix = c.ixy[2 * i];
  const iy = c.ixy[2 * i + 1];
  return {
    index: i,
    cls: c.cls[i],
    className: CLASSES[c.cls[i]]?.name ?? String(c.cls[i]),
    level,
    size,
    xMin: ix * size, xMax: (ix + 1) * size,
    yMin: iy * size, yMax: (iy + 1) * size,
    zMin: c.zCm[3 * i] / 100, zMax: c.zCm[3 * i + 1] / 100, zMean: c.zCm[3 * i + 2] / 100,
    zStd: c.zStdMm[i] / 1000,
    traversability: c.traversability[i] / 255,
    complexity: c.complexity[i] / 255,
    confidence: c.confidence[i] / 255,
    pointCount: c.pointCount[i],
  };
}

/** Point i in metres, sensor frame. */
export function pointXYZ(frame: Frame, i: number): [number, number, number] {
  const p = frame.points.xyzCm;
  return [p[3 * i] / 100, p[3 * i + 1] / 100, p[3 * i + 2] / 100];
}

// ── Precomputed clip ──

export interface ManifestFrame {
  index: number;
  id: string;
  lite: string;               // path relative to manifest.json: grid + every Nth point
  full: string;               // grid + all points (load when paused)
  lite_bytes: number;
  full_bytes: number;
  n_points: number;
  n_cells: number;
  timing_ms: Record<string, number>;
  stats: Record<string, number>;
  accuracy: { accuracy: number; car_iou: number | null; person_iou: number | null } | null;
}

export interface Bookmark {
  label: string;
  frame: string;
  index: number;
  value: number | null;
}

export interface Manifest {
  format: 'LGF1-clip';
  version: number;
  sequence: string;
  fps: number;
  frame_count: number;
  first_frame: string;
  last_frame: string;
  lite_point_stride: number;
  sensor_height_m: number;
  level_sizes_m: number[];
  classes: ClassInfo[];
  bookmarks: Bookmark[];
  summary: Record<string, unknown>;
  frames: ManifestFrame[];
}

export interface LoadedManifest extends Manifest {
  baseUrl: string;            // resolve frame paths against this
}

export async function loadManifest(url: string): Promise<LoadedManifest> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`manifest: ${res.status} ${res.statusText} for ${url}`);
  const m = (await res.json()) as Manifest;
  return { ...m, baseUrl: new URL('.', new URL(url, globalThis.location?.href ?? url)).href };
}

/** URL of a clip frame. kind = 'lite' for playback, 'full' when paused. */
export function frameUrl(manifest: LoadedManifest, index: number, kind: 'lite' | 'full'): string {
  const f = manifest.frames[index];
  return new URL(kind === 'lite' ? f.lite : f.full, manifest.baseUrl).href;
}
