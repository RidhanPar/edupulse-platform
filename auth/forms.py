"""Forms. CSRF protection comes from Flask-WTF."""
from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField
from wtforms.validators import DataRequired, EqualTo, Length

MIN_PASSWORD_LENGTH = 12
# Caps the Argon2 work a single request can force.
MAX_PASSWORD_LENGTH = 256


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired(), Length(max=320)])
    password = PasswordField("Password", validators=[DataRequired(), Length(max=MAX_PASSWORD_LENGTH)])


class SetPasswordForm(FlaskForm):
    password = PasswordField(
        "New password",
        validators=[
            DataRequired(),
            Length(
                min=MIN_PASSWORD_LENGTH,
                max=MAX_PASSWORD_LENGTH,
                message=f"Use between {MIN_PASSWORD_LENGTH} and {MAX_PASSWORD_LENGTH} characters.",
            ),
        ],
    )
    confirm = PasswordField(
        "Confirm new password",
        validators=[DataRequired(), EqualTo("password", message="The passwords do not match.")],
    )


class ChangePasswordForm(SetPasswordForm):
    current_password = PasswordField(
        "Current password", validators=[DataRequired(), Length(max=MAX_PASSWORD_LENGTH)]
    )
