import type {
  HealthResponse,
  DetectionResponse,
  SegmentationResponse,
  ChangeResponse,
  SearchResponse,
  ChatApiRequest,
  ChatApiResponse,
  EvidenceExtractionResponse,
} from '@/types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

class ApiError extends Error {
  status: number;
  data: any;

  constructor(message: string, status: number, data?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let errorMessage = `Server Error (${response.status})`;
    let errorData = null;
    try {
      errorData = await response.json();
      if (errorData?.detail) {
        errorMessage = typeof errorData.detail === 'string' 
          ? errorData.detail 
          : JSON.stringify(errorData.detail);
      }
    } catch {
      // Non-JSON error body fallback
    }
    throw new ApiError(errorMessage, response.status, errorData);
  }
  return response.json() as Promise<T>;
}

export const geoVisionApi = {
  /**
   * Health Check & Subsystem Status
   * GET /health
   */
  async getHealth(signal?: AbortSignal): Promise<HealthResponse> {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
      signal,
    });
    return handleResponse<HealthResponse>(response);
  },

  /**
   * Optical Satellite Object Detection
   * POST /api/v1/detect
   */
  async detectObjects(
    file: File,
    confThresh: number = 0.25,
    signal?: AbortSignal
  ): Promise<DetectionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('conf_thresh', confThresh.toString());

    const response = await fetch(`${API_BASE_URL}/api/v1/detect`, {
      method: 'POST',
      body: formData,
      signal,
    });
    return handleResponse<DetectionResponse>(response);
  },

  /**
   * 5-Class Land-Cover Semantic Segmentation & Area Calculation
   * POST /api/v1/segment
   */
  async segmentLandcover(
    file: File,
    resolutionM: number = 1.0,
    signal?: AbortSignal
  ): Promise<SegmentationResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('resolution_m', resolutionM.toString());

    const response = await fetch(`${API_BASE_URL}/api/v1/segment`, {
      method: 'POST',
      body: formData,
      signal,
    });
    return handleResponse<SegmentationResponse>(response);
  },

  /**
   * Bi-Temporal Change Detection
   * POST /api/v1/change
   */
  async detectChange(
    fileT1: File,
    fileT2: File,
    resolutionM: number = 1.0,
    signal?: AbortSignal
  ): Promise<ChangeResponse> {
    const formData = new FormData();
    formData.append('file_t1', fileT1);
    formData.append('file_t2', fileT2);
    formData.append('resolution_m', resolutionM.toString());

    const response = await fetch(`${API_BASE_URL}/api/v1/change`, {
      method: 'POST',
      body: formData,
      signal,
    });
    return handleResponse<ChangeResponse>(response);
  },

  /**
   * Multimodal Text-to-Image / Image-to-Image Scene Vector Retrieval
   * POST /api/v1/search
   */
  async searchScenes(
    queryText?: string,
    file?: File | null,
    topK: number = 5,
    signal?: AbortSignal
  ): Promise<SearchResponse> {
    const formData = new FormData();
    if (queryText) {
      formData.append('query_text', queryText);
    }
    if (file) {
      formData.append('file', file);
    }
    formData.append('top_k', topK.toString());

    const response = await fetch(`${API_BASE_URL}/api/v1/search`, {
      method: 'POST',
      body: formData,
      signal,
    });
    return handleResponse<SearchResponse>(response);
  },

  /**
   * Grounded VLM Earth Assistant Chat
   * POST /api/v1/chat
   */
  async chatAssistant(
    request: ChatApiRequest,
    signal?: AbortSignal
  ): Promise<ChatApiResponse> {
    const response = await fetch(`${API_BASE_URL}/api/v1/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify(request),
      signal,
    });
    return handleResponse<ChatApiResponse>(response);
  },

  /**
   * Unified Earth Observation Evidence Extraction
   * POST /api/v1/evidence
   */
  async extractEvidence(
    file: File,
    resolutionM: number = 1.0,
    signal?: AbortSignal
  ): Promise<EvidenceExtractionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('resolution_m', resolutionM.toString());

    const response = await fetch(`${API_BASE_URL}/api/v1/evidence`, {
      method: 'POST',
      body: formData,
      signal,
    });
    return handleResponse<EvidenceExtractionResponse>(response);
  },
};

export { ApiError };
export default geoVisionApi;
