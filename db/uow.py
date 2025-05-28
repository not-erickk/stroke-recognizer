from typing import Callable
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import os
from dotenv import load_dotenv

load_dotenv()

class UnitOfWork:
    def init(self):
        # Load configuration from .env file
        if not load_dotenv():
            raise EnvironmentError("Failed to load .env file")

        # Get database configuration from .env file
        try:
            DB_USER = os.environ['DB_USER']
            DB_PASSWORD = os.environ['DB_PASSWORD']
            DB_HOST = os.environ['DB_HOST']
            DB_NAME = os.environ['DB_NAME']
        except KeyError as e:
            raise EnvironmentError(f"Missing required environment variable: {e}")

        # Create database URL
        DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

        # Create engine and session factory
        self.engine = create_engine(DATABASE_URL)
        self.SessionFactory = sessionmaker(bind=self.engine)

    @contextmanager
    def start(self):
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def execute(self, fn: Callable[[Session], any]):
        """
        Execute a function within a transaction context

        Args:
            fn: A function that takes a session as parameter and performs database operations

        Returns:
            The result of the executed function
        """
        with self.start() as session:
            result = fn(session)
            return result

    def bulk_execute(self, operations: list[Callable[[Session], any]]):
        """
        Execute multiple operations in a single transaction

        Args:
            operations: List of functions that take a session as parameter

        Returns:
            List of results from all operations
        """
        with self.start() as session:
            results = []
            for operation in operations:
                results.append(operation(session))
            return results