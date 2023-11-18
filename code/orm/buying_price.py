from datetime import datetime, timedelta

from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    String,
    Date,
    Numeric,
    ForeignKey,
    and_,
)
from sqlalchemy.orm import validates
from sqlalchemy.sql.expression import func, true

from models.database import Base, db_session


class BuyingPriceMarket(Base):
    __tablename__ = "buying_price_market"
    id = Column(Integer, primary_key=True)
    active = Column(Boolean)
    id_market_step = Column(
        Integer, ForeignKey("buying_price_market_step.id", ondelete="CASCADE")
    )
    id_market_type = Column(
        Integer, ForeignKey("buying_price_market_type.id", ondelete="CASCADE")
    )
    market_name = Column(String(255))
    key = Column(String(255))

    def __init__(
        self, active, id_market_step, id_market_type, market_name, key
    ):
        self.active = active
        self.id_market_step = id_market_step
        self.id_market_type = id_market_type
        self.market_name = market_name
        self.key = key

    @classmethod
    def get_key(cls, id_market):
        key = db_session.query(cls.key).filter_by(id=id_market).first()
        return key[0] if key is not None else None

    @classmethod
    def get_one(cls, id_market=1):
        return db_session.query(cls).filter_by(id=id_market)

    @classmethod
    def get_all(cls, active=True):
        return db_session.query(cls.id, cls.key, cls.market_name).filter_by(
            active=active
        )

    @classmethod
    def add_all(cls, markets):
        db_session.add_all(markets)


class BuyingPriceMarketDate(Base):
    __tablename__ = "buying_price_market_date"
    id = Column(Integer, primary_key=True)
    id_market = Column(
        Integer, ForeignKey("buying_price_market.id", ondelete="CASCADE")
    )
    market_date = Column(Date)
    scrapped = Column(Boolean)

    def __init__(self, id_market, market_date, scrapped):
        self.id_market = id_market
        self.market_date = market_date
        self.scrapped = scrapped

    @classmethod
    def get_all_not_scrapped_for_market(cls, id_market):
        return db_session.query(cls).filter_by(
            id_market=id_market, scrapped=False
        )

    @classmethod
    def get_all_not_scrapped(cls):
        return (
            db_session.query(
                cls.id,
                cls.market_date,
                cls.id_market,
                BuyingPriceMarket.key,
                BuyingPriceMarket.id_market_step,
                BuyingPriceMarket.id_market_type,
            )
            .join(BuyingPriceMarket)
            .filter(BuyingPriceMarket.active == true())
            .filter(cls.scrapped != true())
            .order_by(cls.market_date.desc())
        )

    @classmethod
    def get_id(cls, id_market, date):
        id_date = (
            db_session.query(cls.id)
            .filter_by(id_market=id_market, market_date=date)
            .first()
        )
        return id_date[0] if id_date is not None else None

    @classmethod
    def get_max_date(cls, id_market):
        id_date = (
            db_session.query(func.max(cls.market_date))
            .filter_by(id_market=id_market)
            .first()
        )
        return id_date[0] if id_date is not None else None


class BuyingPriceMarketStep(Base):
    __tablename__ = "buying_price_market_step"
    id = Column(Integer, primary_key=True)
    active = Column(Boolean)
    key = Column(String(255))
    market_step_name = Column(String(255))

    def __init__(self, active, key, market_step_name):
        self.active = active
        self.key = key
        self.market_step_name = market_step_name

    @classmethod
    def get_key(cls, id_market_step):
        key = db_session.query(cls.key).filter_by(id=id_market_step).first()
        return key[0] if key is not None else None


class BuyingPriceMarketType(Base):
    __tablename__ = "buying_price_market_type"
    id = Column(Integer, primary_key=True)
    active = Column(Boolean)
    key = Column(String(255))
    market_type_name = Column(String(255))

    def __init__(self, active, key, market_type_name):
        self.active = active
        self.key = key
        self.market_type_name = market_type_name

    @classmethod
    def get_key(cls, id_market_type):
        key = db_session.query(cls.key).filter_by(id=id_market_type).first()
        return key[0] if key is not None else None


class BuyingPriceMarketStepType(Base):
    __tablename__ = "buying_price_market_step_type"
    id = Column(Integer, primary_key=True)
    active = Column(Boolean)
    id_market_step = Column(
        Integer, ForeignKey("buying_price_market_step.id", ondelete="CASCADE")
    )
    id_market_type = Column(
        Integer, ForeignKey("buying_price_market_type.id", ondelete="CASCADE")
    )

    def __init__(self, active, id_market_step, id_market_type):
        self.active = active
        self.id_market_step = id_market_step
        self.id_market_type = id_market_type

    @classmethod
    def get_active(cls):
        return (
            db_session.query(cls.id, cls.id_market_step, cls.id_market_type)
            .join(BuyingPriceMarketStep)
            .join(BuyingPriceMarketType)
            .filter(BuyingPriceMarketStepType.active == true())
            .filter(BuyingPriceMarketStep.active == true())
            .filter(cls.active == true())
        )


class BuyingPriceProduct(Base):
    __tablename__ = "buying_price_product"
    id = Column(Integer, primary_key=True)
    active = Column(Boolean)
    avg_price = Column(Numeric(10, 3))
    date_price = Column(Date)
    id_market = Column(
        Integer, ForeignKey("buying_price_market.id", ondelete="CASCADE")
    )
    id_market_step = Column(
        Integer, ForeignKey("buying_price_market_step.id", ondelete="CASCADE")
    )
    id_market_type = Column(
        Integer, ForeignKey("buying_price_market_type.id", ondelete="CASCADE")
    )
    max_price = Column(Numeric(10, 3))
    min_price = Column(Numeric(10, 3))
    product_name = Column(String(255))
    unit = Column(String(255))

    @validates("id_market", "id_market_type")
    def validate_id(self, key, id_value):
        assert 0 <= id_value <= 9999, f"{key} must be between 0 and 9999"
        return id_value

    @validates("avg_price", "min_price", "max_price")
    def validate_price(self, key, price):
        assert price >= 0, f"{key} must be a positive number"
        return price

    def __init__(
        self,
        active,
        avg_price,
        date_price,
        id_market,
        id_market_step,
        id_market_type,
        max_price,
        min_price,
        product_name,
        unit,
    ):
        self.active = active
        self.avg_price = avg_price
        self.date_price = date_price
        self.id_market = id_market
        self.id_market_step = id_market_step
        self.id_market_type = id_market_type
        self.max_price = max_price
        self.min_price = min_price
        self.product_name = product_name
        self.unit = unit

    @classmethod
    def get_product_for_market(cls, market_name, product_name):
        return (
            db_session.query(
                cls.date_price,
                cls.product_name,
                cls.avg_price,
                BuyingPriceMarket.market_name,
            )
            .join(BuyingPriceMarket)
            .filter(BuyingPriceMarket.market_name == market_name)
            .filter(cls.product_name == product_name)
            .order_by(cls.date_price, cls.product_name)
            .all()
        )

    @classmethod
    def get_products_name_for_market(cls, market_name):
        return (
            db_session.query(cls.product_name)
            .distinct()
            .join(BuyingPriceMarket)
            .filter(BuyingPriceMarket.market_name == market_name)
            .order_by(cls.product_name)
            .all()
        )

    @classmethod
    def get_products(cls):
        return (
            db_session.query(cls.product_name, cls.avg_price)
            .distinct()
            .order_by(cls.product_name)
            .all()
        )

    @classmethod
    def get_products_like(cls, user_input, user_days):
        # Construct a pattern string with '%' on both sides of the user_input
        # '%' is a wildcard character that matches any number of characters in SQL like operations
        pattern = f"%{user_input}%"
        n_days_ago = datetime.utcnow() - timedelta(days=user_days)
        subquery = (
            db_session.query(
                cls.product_name, func.min(cls.date_price).label("min_date")
            )
            .filter(
                and_(
                    cls.date_price >= n_days_ago,
                    cls.product_name.ilike(pattern),
                )
            )
            .group_by(cls.product_name)
            .subquery()
        )

        return (
            db_session.query(cls)
            .join(
                subquery,
                and_(
                    cls.product_name == subquery.c.product_name,
                    cls.date_price == subquery.c.min_date,
                ),
            )
            .all()
        )

    @classmethod
    def get_latest_products_prices_and_names(cls):
        subquery = (
            db_session.query(
                cls.product_name, func.max(cls.date_price).label("last_date")
            )
            .group_by(cls.product_name)
            .subquery()
        )

        return (
            db_session.query(cls.product_name, cls.avg_price)
            .join(
                subquery,
                and_(
                    cls.product_name == subquery.c.product_name,
                    cls.date_price == subquery.c.last_date,
                ),
            )
            .all()
        )
