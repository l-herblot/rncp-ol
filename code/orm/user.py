import secrets
import string

from sqlalchemy import Boolean, Column, Integer, String

from models.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    admin = Column(Boolean, default=False)
    token = Column(String)
    quota = Column(Integer)
    n = Column(Integer)

    def __init__(self, name, quota: int, n: int, admin: bool = False):
        self.token = generate_token()
        self.name = name
        self.quota = quota
        self.n = n
        self.admin = admin


def generate_token(length=16):
    alphabet = string.ascii_letters + string.digits
    token = "".join(secrets.choice(alphabet) for _ in range(length))
    return token
