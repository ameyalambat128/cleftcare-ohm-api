import time
import requests
from typing import Dict, Any, Optional


class CallbackService:
    """Service for sending callback notifications with retry logic."""

    def __init__(self, max_retries: int = 5, timeout: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout

    def send_callback(
        self,
        url: str,
        payload: Dict[str, Any],
        request_id: str,
    ) -> bool:
        """
        Send callback POST request with exponential backoff retry logic.

        Args:
            url: Callback URL to POST results to
            payload: JSON payload to send
            request_id: Request ID for logging

        Returns:
            True if callback succeeded, False otherwise
        """
        retry_delays = [1, 2, 4, 8, 16]  # Exponential backoff in seconds

        for attempt in range(self.max_retries):
            try:
                print(
                    f"[{request_id}] Attempting callback {attempt + 1}/{self.max_retries} "
                    f"to {url}"
                )

                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"},
                )

                if response.status_code >= 200 and response.status_code < 300:
                    print(
                        f"[{request_id}] Callback succeeded on attempt {attempt + 1} "
                        f"(status: {response.status_code})"
                    )
                    return True
                else:
                    print(
                        f"[{request_id}] Callback failed with status {response.status_code} "
                        f"on attempt {attempt + 1}: {response.text[:200]}"
                    )

            except requests.exceptions.Timeout:
                print(
                    f"[{request_id}] Callback timeout on attempt {attempt + 1}/{self.max_retries}"
                )
            except requests.exceptions.ConnectionError as e:
                print(
                    f"[{request_id}] Callback connection error on attempt {attempt + 1}/{self.max_retries}: {str(e)}"
                )
            except requests.exceptions.RequestException as e:
                print(
                    f"[{request_id}] Callback request error on attempt {attempt + 1}/{self.max_retries}: {str(e)}"
                )
            except Exception as e:
                print(
                    f"[{request_id}] Unexpected callback error on attempt {attempt + 1}/{self.max_retries}: {str(e)}"
                )

            # Wait before retry (except on last attempt)
            if attempt < self.max_retries - 1:
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                print(f"[{request_id}] Waiting {delay}s before retry...")
                time.sleep(delay)

        print(
            f"[{request_id}] Callback failed after {self.max_retries} attempts. "
            f"URL: {url}"
        )
        return False

