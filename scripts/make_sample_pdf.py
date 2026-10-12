"""Generates the sample PDF used in the demo and the retrieval evaluation.

Northwind Labs is a made-up company. The handbook is written so that each policy
lives on a specific page, which makes it easy to check whether citations are right.
"""
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

PAGES = [
    ("Welcome and working hours", [
        "Welcome to Northwind Labs. This handbook explains the policies that apply to all full-time employees. "
        "Contractors should check their contract first, because some of these policies do not apply to them.",
        "Core working hours are 11:00 to 16:00 Pakistan Standard Time, Monday to Friday. Outside core hours you "
        "can plan your day as you like, as long as you work 40 hours a week on average.",
        "Every team holds a 15-minute stand-up at 11:15. If you cannot attend, post your update in the team "
        "channel before 11:15.",
        "Public holidays follow the official government calendar. If a holiday falls on a weekend, it is not moved "
        "to a weekday.",
    ]),
    ("Leave policy", [
        "Full-time employees get 20 days of paid annual leave per calendar year. Leave is earned at 1.67 days per "
        "month, so new joiners get a pro-rated amount in their first year.",
        "Up to 5 unused annual leave days can be carried over to the next year. Carried-over days must be used "
        "before 31 March, or they are lost.",
        "Sick leave is 10 paid days per year. If you are off sick for more than 2 consecutive days, you need to "
        "share a doctor's note with HR.",
        "Annual leave requests must be submitted in the HR portal at least 2 weeks in advance for anything longer "
        "than 3 days. Shorter leave needs 2 working days of notice.",
        "Parental leave is 12 weeks fully paid for the primary caregiver and 4 weeks fully paid for the secondary "
        "caregiver.",
    ]),
    ("Remote work", [
        "Northwind Labs is remote-first. You can work from home or from any location in Pakistan with a stable "
        "connection of at least 20 Mbps.",
        "Working from outside Pakistan is allowed for up to 30 days per calendar year and needs written approval "
        "from your manager and HR at least 3 weeks before you travel.",
        "Each new employee receives a one-time home office allowance of PKR 60,000 for a desk, chair or monitor. "
        "Receipts must be submitted within 60 days of joining.",
        "Internet costs are reimbursed up to PKR 5,000 per month. Submit the bill through the expense system.",
        "Once a quarter every team meets in person in Karachi for two days. Travel and hotel are paid by the company.",
    ]),
    ("Expenses and travel", [
        "Business expenses must be submitted in the expense system within 30 days of the purchase, with a clear "
        "photo of the receipt.",
        "Any single expense above PKR 25,000 needs your manager's approval before you spend the money.",
        "For business travel, book economy class for flights shorter than 6 hours. Flights of 6 hours or more can be "
        "booked in premium economy.",
        "The daily meal allowance while travelling is PKR 4,000 within Pakistan and USD 50 abroad. Alcohol is never "
        "reimbursed.",
        "Approved expenses are paid with the next salary run. Salaries are paid on the last working day of each month.",
    ]),
    ("Information security", [
        "All laptops must use full-disk encryption and lock automatically after 5 minutes of inactivity.",
        "Two-factor authentication is mandatory for email, the code repository and the HR portal. Use an "
        "authenticator app, not SMS.",
        "Never share customer data over personal email or messaging apps. Customer data can only be stored in the "
        "approved cloud storage.",
        "Report a lost or stolen device to the IT helpdesk at security@northwindlabs.example within 1 hour.",
        "Passwords must be at least 14 characters long and stored in the company password manager.",
    ]),
    ("Performance and growth", [
        "Performance reviews happen twice a year, in June and December. Each review covers impact, collaboration "
        "and growth.",
        "Salary adjustments are decided in the December review and take effect from 1 January.",
        "Every employee has a yearly learning budget of PKR 100,000 for courses, books and conferences. Unused "
        "budget does not carry over.",
        "Employees can apply for an internal transfer after 12 months in their current role.",
        "The probation period for new employees is 3 months. During probation, the notice period is 2 weeks; "
        "after probation, it is 1 month.",
    ]),
]


def build(path: Path) -> None:
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(path), pagesize=A4, title="Northwind Labs Employee Handbook")
    story = []
    for i, (title, paras) in enumerate(PAGES):
        story.append(Paragraph(f"{i + 1}. {title}", styles["Heading1"]))
        story.append(Spacer(1, 8))
        for p in paras:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 6))
        if i < len(PAGES) - 1:
            story.append(PageBreak())
    doc.build(story)


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "sample_docs" / "northwind_handbook.pdf"
    build(out)
    print("wrote", out)
