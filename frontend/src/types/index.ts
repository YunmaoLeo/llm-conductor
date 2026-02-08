export interface MusicFeatures {
  note_density: number;
  note_count: number;
  duration_seconds: number;
  pitch_range: [number, number];
  pitch_mean: number;
  pitch_std: number;
  onset_density_curve: number[];
  instruments_used: number[];
  instrument_note_counts: Record<string, number>;
  has_excessive_notes: boolean;
  silence_ratio: number;
}

export interface WsMessage {
  type: 'status' | 'plan' | 'tokens' | 'evaluation' | 'complete' | 'error';
  message?: string;
  iteration?: number;
  max_iterations?: number;
  // plan
  plan?: Record<string, unknown>;
  // tokens
  token_count?: number;
  generation_time?: number;
  text_description?: string;
  // evaluation
  score?: number;
  verdict?: string;
  strengths?: string[];
  weaknesses?: string[];
  suggestions?: string[];
  // complete
  success?: boolean;
  generation_id?: string;
  midi_url?: string;
  features?: MusicFeatures;
  iterations?: number;
}

export type CompositionStatus =
  | 'idle'
  | 'connecting'
  | 'generating'
  | 'evaluating'
  | 'complete'
  | 'error';
