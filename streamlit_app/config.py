"""
Config file for Streamlit App
"""

from member import Member



TITLE = "Hybrid Movie Recommendation"


TEAM_MEMBERS = [
    Member(
        name="Michèle Dubuisson",
        linkedin_url="https://www.linkedin.com/in/micheledubuisson/",
        github_url="https://github.com/MicDfr",
    ),
    Member(
        name="Thomas Sabatier",
        linkedin_url="https://www.linkedin.com/in/thomas-sabatier0/",
        github_url="https://github.com/ThomasSabatier",
    ),
    Member(
        name="Antoine Sibille",
        linkedin_url="https://www.linkedin.com/in/antoinesibille/",
        github_url="https://github.com/asib-fr",
    )
]

PROMOTION = "Promotion Bootcamp Data Scientist - January 2023"