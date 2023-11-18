from sqlalchemy import Column, Integer, String

from models.database import db_session, Base


class ReferenceIndex(Base):
    __tablename__ = "reference_index"
    id = Column(Integer, primary_key=True)
    reference_type = Column(String(255))
    custom_name = Column(String(255))

    def __init__(self, reference_type, custom_name):
        self.reference_type = reference_type
        self.custom_name = custom_name

    @classmethod
    def get_reference_list(cls, input_name):
        """select custom_name where reference_type = input_name"""
        return (
            db_session.query(cls.custom_name)
            .distinct()
            .order_by(cls.custom_name)
            .filter_by(reference_type=input_name)
            .all()
        )


class MarketNameIngredient(Base):
    __tablename__ = "market_name_ingredients_assoc"
    id = Column(Integer, primary_key=True)
    reference_name = Column(String(255))
    custom_name = Column(String(255))

    def __init__(self, custom_name, reference_name):
        self.reference_name = reference_name
        self.custom_name = custom_name

    @classmethod
    def get_ingredients_references(cls):
        return (
            db_session.query(cls.custom_name, cls.reference_name)
            .distinct()
            .order_by(cls.custom_name)
            .all()
        )
