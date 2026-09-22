/**
 * TypeScript type definitions strictly mirroring FastAPI Pydantic schemas in geovision/api/schemas.py
 */

export interface HealthResponse {
  status: string;
  version: string;
  device: string;
  models_status: Record<string, boolean>;
}

export interface BBoxItem {
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  confidence: number;
  class_id: number;
  class_name: string;
}

export interface DetectionResponse {
  count: number;
  detections: BBoxItem[];
  image_shape: [number, number, number];
  latency_ms: number;
}

export interface ClassBreakdown {
  class_id: number;
  class_name: string;
  pixel_count: number;
  percentage: number;
  area_hectares: number;
  area_sqkm: number;
}

export interface SegmentationResponse {
  mask_shape: [number, number];
  classes: ClassBreakdown[];
  resolution_m: number;
  total_area_hectares: number;
  latency_ms: number;
}

export interface ChangeResponse {
  changed_pixels: number;
  total_pixels: number;
  change_ratio: number;
  changed_area_ha: number;
  resolution_m: number;
  latency_ms: number;
}

export interface SearchResultItem {
  id: string | number;
  score: number;
  label: string | null;
  metadata: Record<string, any>;
}

export interface SearchResponse {
  query: string;
  count: number;
  results: SearchResultItem[];
  latency_ms: number;
}

export interface ChatApiRequest {
  query: string;
  context_evidence?: Record<string, any> | null;
  provider?: string;
  api_key?: string | null;
}

export interface ChatApiResponse {
  query: string;
  answer: string;
  citations: string[];
  grounding_score: number;
  ungrounded_claims: string[];
  is_grounded: boolean;
  latency_ms: number;
}

export interface EvidenceExtractionResponse {
  timestamp: string;
  evidence_tags: string[];
  summary_text: string;
  raw_evidence: Record<string, any>;
  latency_ms: number;
}

export type ApiStatus = 'idle' | 'loading' | 'success' | 'error';
