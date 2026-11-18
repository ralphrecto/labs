export interface ScanPoints { count: number; positions: Float32Array /* xyz, meters */; colors?: Float32Array /* rgb 0..1 */ }
export interface ScanData { id: string; name: string; source: string /* 'scaniverse' | 'ply' | 'synthetic' ... */; points: ScanPoints; upAxis: 'y' | 'z' /* default y */; metric: boolean /* true once scale is set */ }
