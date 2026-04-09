import os

from mailjet_rest import Client


EMAIL_RECIPIENTS = [
    {
        "Email": "alambat@asu.edu",
        "Name": "Cleft Care Admin",
    }
]
DASHBOARD_BASE_URL = "https://cleftcare-dashboard.vercel.app/dashboard"


def _mailjet_client():
    return Client(
        auth=(
            os.getenv("MAILJET_API_KEY"),
            os.getenv("MAILJET_API_SECRET"),
        ),
        version="v3.1",
    )


def _send_email(subject, body):
    data = {
        "Messages": [
            {
                "From": {
                    "Email": os.getenv("EMAIL_FROM"),
                    "Name": os.getenv("EMAIL_FROM_NAME"),
                },
                "To": EMAIL_RECIPIENTS,
                "Subject": subject,
                "HTMLPart": body,
            }
        ]
    }

    try:
        result = _mailjet_client().send.create(data=data)
        if result.status_code == 200:
            print(f"Email sent successfully: {result.json()}")
        else:
            print(f"Failed to send email: {result.status_code}, {result.json()}")
    except Exception as exc:
        print(f"Exception occurred while sending email: {exc}")


def _dashboard_link(user_id):
    return f"{DASHBOARD_BASE_URL}/{user_id}"


def send_test_prediction_email(user_id, name, community_worker_name, perceptual_rating):
    body = (
        f"<p>The patient {user_id} and {name} was recorded in ASU by the community worker {community_worker_name}.</p>"
        f"<p>The level of hypernasality in its speech as per cleft care is {perceptual_rating}.</p>"
    )
    _send_email("CleftCare: New Prediction Result", body)


def send_ohm_result_email(user_id, name, community_worker_name):
    body = (
        f"<p>The patient {name} recorded by the community worker {community_worker_name}.</p>"
        f"<p>The link to the more details for the patient - {_dashboard_link(user_id)}.</p>"
    )
    _send_email("CleftCare: New Prediction Result", body)


def send_gop_result_email(user_id, name, community_worker_name, sentence_gop):
    body = (
        f"<p>The patient {name} recorded by the community worker {community_worker_name}.</p>"
        f"<p>GOP Assessment completed with sentence score: {sentence_gop}</p>"
        f"<p>The link to the more details for the patient - {_dashboard_link(user_id)}.</p>"
    )
    _send_email("CleftCare: New GOP Assessment Result", body)
