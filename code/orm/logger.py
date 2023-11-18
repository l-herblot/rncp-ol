from sqlalchemy import Column, Date, Integer, String, Text, TIMESTAMP
from models.database import Base, db_session


class Event:
    """This class avoids many duplicates between its children but won't be used on its own.
    You guessed right : its children are LoggingEvent and QueueEvent !
    """

    id = Column(Integer, primary_key=True)
    event_time = Column(TIMESTAMP)
    event_type = Column(String(20))
    logger_id = Column(Text)
    trigger = Column(Text)
    message = Column(Text)

    def __init__(self, event_time, event_type, logger_id, trigger, message):
        self.event_time = event_time
        self.event_type = event_type
        self.logger_id = logger_id
        self.trigger = trigger
        self.message = message

    @staticmethod
    def decapsulate_events(events):
        results = []
        for event in events:
            results.append(
                {
                    "id": event.id,
                    "event_time": event.event_time,
                    "event_type": event.event_type,
                    "logger_id": event.logger_id,
                    "trigger": event.trigger,
                    "message": event.message,
                }
            )
        return results

    @classmethod
    def delete_events(cls, caller_id, loggers_ids, event_dates, event_types):
        """Obviously delete events from the database
        caller_id should be a string
        logger_ids should be a list
        event_dates should be a string or a list of length one or two strings (with date format yyyy-mm-dd)
        event_types should be a list
        """
        cond = [cls.logger_id.in_(loggers_ids)]

        if caller_id != "":
            cond.append(cls.trigger.like(f"%::{caller_id}"))

        if isinstance(event_dates, list):
            if len(event_dates) == 2:
                cond.append(
                    cls.event_time.between(event_dates[0], event_dates[1])
                )
            elif len(event_dates) == 1:
                cond.append(cls.event_time.cast(Date) == event_dates[0])
        elif event_dates != "":
            cond.append(cls.event_time.cast(Date) == event_dates)

        if len(event_types) > 0:
            cond.append(cls.event_type.in_(event_types))

        db_session.query(cls).filter(*cond).delete()

        db_session.commit()

    @classmethod
    def get_events(
        cls,
        caller_id,
        loggers_ids,
        event_dates,
        event_types,
        decapsulate=False,
    ):
        """Return all corresponding events
        caller_id should be a string
        logger_ids should be a list
        event_dates should be a string or a list of length one or two (with date format yyyy-mm-dd)
        event_types should be a list
        decapsulate specifies if the result should be LoggingEvent/QueueEvent objects or a list of dictionaries
        """
        cond = [cls.logger_id.in_(loggers_ids)]

        if caller_id != "":
            cond.append(cls.trigger.like(f"%::{caller_id}"))

        if isinstance(event_dates, list):
            if len(event_dates) == 2:
                cond.append(
                    cls.event_time.between(event_dates[0], event_dates[1])
                )
            elif len(event_dates) == 1:
                cond.append(cls.event_time.cast(Date) == event_dates[0])
        elif event_dates != "":
            cond.append(cls.event_time.cast(Date) == event_dates)

        if len(event_types) > 0:
            cond.append(cls.event_type.in_(event_types))

        events = (
            db_session.query(cls).filter(*cond).order_by(cls.event_time.desc())
        )

        return (
            cls.decapsulate_events(events) if decapsulate is True else events
        )

    @classmethod
    def get_loggers(cls):
        """Return every logger id"""
        return (
            db_session.query(cls.logger_id)
            .group_by(cls.logger_id)
            .order_by(cls.logger_id.asc())
        )


class LoggingEvent(Event, Base):
    __tablename__ = "logging"

    def __init__(self, event_time, event_type, logger_id, trigger, message):
        super(LoggingEvent, self).__init__(
            event_time, event_type, logger_id, trigger, message
        )

    @classmethod
    def add(cls, time, level, logger_id, trigger, message):
        """Obviously save an event to the database"""
        db_session.add(
            cls(
                time,
                level,
                logger_id,
                trigger,
                message,
            )
        )

        db_session.commit()


class QueueEvent(Event, Base):
    __tablename__ = "logging_queue"

    def __init__(
        self,
        event_time,
        event_type,
        logger_id,
        trigger,
        message,
    ):
        super(QueueEvent, self).__init__(
            event_time, event_type, logger_id, trigger, message
        )

    @classmethod
    def add(cls, time, level, logger_id, trigger, message):
        """Obviously save an event to the database"""
        db_session.add(
            cls(
                time,
                level,
                logger_id,
                trigger,
                message,
            )
        )

        db_session.commit()

    @classmethod
    def get_queue(cls, decapsulate=False):
        """Return every event left in the queue
        decapsulate specifies if the result should be QueueEvent objects or a list of dictionaries
        """
        results = db_session.query(
            cls.id,
            cls.event_time,
            cls.event_type,
            cls.logger_id,
            cls.trigger,
            cls.message,
        ).order_by(
            cls.event_time.cast(Date).desc(),
            cls.logger_id.asc(),
            cls.trigger.asc(),
        )

        return (
            cls.decapsulate_events(results) if decapsulate is True else results
        )

    @classmethod
    def wipe_queue(cls):
        """Wipe the queue, deleting every remaining entry"""
        db_session.query(cls).delete()
        db_session.commit()
