import json
import math
import numbers
import uuid
from typing import Any, Dict, Optional

from fastapi.encoders import jsonable_encoder

from gop_regression import predict_gop_regression_score
from .supabase_client import get_supabase_client


class SupabaseSync:
    """Wrapper that persists OHM/GOP results to Supabase when configured."""

    def __init__(self) -> None:
        self.client = get_supabase_client()

    def upsert_user_audio_file(
        self,
        *,
        user_id: str,
        prompt: Optional[str],
        prompt_number: Optional[int],
        s3_key: Optional[str],
        language: Optional[str],
        gop_sentence_score: Optional[float],
        ohm_score: Optional[float],
        all_gop_scores: Optional[Dict[str, Any]],
        per_phone_gop: Optional[Any],
        request_id: str,
        processing_time: float,
    ) -> None:
        if self.client is None:
            return

        sanitized_all_gop = self._sanitize_json_field(all_gop_scores)
        sanitized_per_phone = self._sanitize_json_field(per_phone_gop)

        payload: Dict[str, Any] = {
            "userId": user_id,
            "prompt": prompt,
            "promptNumber": prompt_number,
            "s3Key": s3_key,
            "language": language,
            "gopSentenceScore": self._safe_float(gop_sentence_score),
            "ohmScore": self._safe_float(ohm_score),
            "allGopScores": sanitized_all_gop,
            "perPhoneGop": sanitized_per_phone,
            "requestId": request_id,
            "processingTime": processing_time,
        }

        normalized_payload = self._normalize_payload(payload)
        sanitized_payload = self._sanitize_json_field(normalized_payload)
        if not isinstance(sanitized_payload, dict):
            raise ValueError("Supabase payload sanitization failed.")

        try:
            self.client.table("UserAudioFile").upsert(sanitized_payload).execute()
            print(f"Supabase UserAudioFile upserted for request {request_id}")
        except Exception as exc:
            print(f"Supabase upsert failed for request {request_id}: {exc}")
            print(f"Debug - allGopScores type: {type(sanitized_all_gop)}, value preview: {str(sanitized_all_gop)[:200]}")
            print(f"Debug - perPhoneGop type: {type(sanitized_per_phone)}, value preview: {str(sanitized_per_phone)[:200]}")
            return

        # Update user average scores after successful upsert
        self._update_user_average_scores(user_id=user_id)

    def _update_user_average_scores(self, *, user_id: str) -> None:
        """Recalculate and update average OHM/GOP scores for a user."""
        if not self.client or not user_id:
            return

        try:
            # Query all scores for this user
            response = (
                self.client.table("UserAudioFile")
                .select("ohmScore, gopSentenceScore")
                .eq("userId", user_id)
                .execute()
            )

            records = response.data or []

            # Calculate averages (exclude None/null values)
            ohm_scores = [r["ohmScore"] for r in records if r.get("ohmScore") is not None]
            gop_scores = [r["gopSentenceScore"] for r in records if r.get("gopSentenceScore") is not None]

            average_ohm = sum(ohm_scores) / len(ohm_scores) if ohm_scores else None
            average_gop = predict_gop_regression_score(gop_scores)
            if average_gop is None:
                average_gop = sum(gop_scores) / len(gop_scores) if gop_scores else None

            # Update User table
            self.client.table("User").update({
                "averageOHMScore": average_ohm,
                "averageGOPScore": average_gop,
            }).eq("id", user_id).execute()

            print(f"Updated User {user_id} averages: OHM={average_ohm}, GOP={average_gop}")

        except Exception as exc:
            print(f"Failed to update user average scores for {user_id}: {exc}")

    @staticmethod
    def _normalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload)

        record_id = normalized.get("id")
        if record_id:
            normalized["id"] = str(record_id)
        else:
            request_id = normalized.get("requestId")
            normalized["id"] = str(request_id) if request_id else str(uuid.uuid4())

        prompt = normalized.get("prompt")
        if prompt is None:
            normalized["prompt"] = ""
        else:
            normalized["prompt"] = str(prompt)

        if normalized.get("promptNumber") is None:
            normalized["promptNumber"] = 0
        else:
            try:
                normalized["promptNumber"] = int(normalized["promptNumber"])
            except (TypeError, ValueError):
                normalized["promptNumber"] = 0

        language = normalized.get("language")
        if language is None or str(language).strip() == "":
            normalized["language"] = "unknown"
        else:
            normalized["language"] = str(language)

        s3_key = normalized.get("s3Key")
        if s3_key is None or str(s3_key).strip() == "":
            normalized["s3Key"] = "unknown"
        else:
            normalized["s3Key"] = str(s3_key)

        processing_time = normalized.get("processingTime")
        if processing_time is not None:
            try:
                normalized["processingTime"] = int(max(0, round(float(processing_time) * 1000)))
            except (TypeError, ValueError):
                normalized["processingTime"] = None

        return normalized

    @staticmethod
    def _safe_float(value: Optional[Any]) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, bool):
            return float(value)

        if isinstance(value, numbers.Real):
            numeric_value = float(value)
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                return None
            return numeric_value

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(numeric_value) or math.isinf(numeric_value):
            return None
        return numeric_value

    @classmethod
    def _sanitize_json_field(cls, value: Optional[Any]) -> Optional[Any]:
        if value is None:
            return None

        if isinstance(value, bool):
            return value

        if isinstance(value, (str,)):
            return value

        if isinstance(value, numbers.Integral):
            return int(value)

        if isinstance(value, numbers.Real):
            return cls._safe_float(value)

        if hasattr(value, "item"):
            try:
                return cls._sanitize_json_field(value.item())
            except Exception:
                pass

        if isinstance(value, (list, tuple, set)):
            return [cls._sanitize_json_field(item) for item in value]

        if hasattr(value, "tolist"):
            try:
                return cls._sanitize_json_field(value.tolist())
            except Exception:
                pass

        if isinstance(value, dict):
            return {key: cls._sanitize_json_field(val) for key, val in value.items()}

        try:
            return jsonable_encoder(value)
        except Exception:
            return jsonable_encoder(str(value))
