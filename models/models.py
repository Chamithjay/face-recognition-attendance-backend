"""
Database models for Face Recognition Attendance System - Simplified Version
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base


class Student(Base):
    """
    Student model for storing student information.
    
    Attributes:
        id: Primary key
        student_id: Unique student registration number
        name: Full name of the student
        email: Unique email address
        created_at: Timestamp when record was created
    """
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(200), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    attendance_records = relationship("AttendanceRecord", back_populates="student")

    def __repr__(self):
        return f"<Student(id={self.id}, student_id='{self.student_id}', name='{self.name}')>"


class ClassSession(Base):
    """
    Class session model for storing class information.
    
    Attributes:
        id: Primary key
        class_name: Name of the class
        start_time: Scheduled start time
        end_time: Scheduled end time
        is_active: Whether class is currently taking attendance
        created_at: Timestamp when record was created
    """
    __tablename__ = "class_sessions"

    id = Column(Integer, primary_key=True, index=True)
    class_name = Column(String(200), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    attendance_records = relationship("AttendanceRecord", back_populates="class_session")

    def __repr__(self):
        return f"<ClassSession(id={self.id}, class_name='{self.class_name}')>"


class AttendanceRecord(Base):
    """
    Attendance record model for storing attendance information.
    
    Attributes:
        id: Primary key
        student_id: Foreign key to students table
        class_session_id: Foreign key to class_sessions table
        check_in_time: When student was detected/checked in
        confidence_score: Face recognition confidence score (0.0 to 1.0)
        created_at: Timestamp when record was created
    """
    __tablename__ = "attendance_records"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    class_session_id = Column(Integer, ForeignKey("class_sessions.id"), nullable=False)
    check_in_time = Column(DateTime(timezone=True), server_default=func.now())
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    student = relationship("Student", back_populates="attendance_records")
    class_session = relationship("ClassSession", back_populates="attendance_records")

    def __repr__(self):
        return f"<AttendanceRecord(id={self.id}, student_id={self.student_id}, class_session_id={self.class_session_id})>"


class User(Base):
    """
    User model for storing authentication information.
    
    Attributes:
        id: Primary key
        username: Unique username for login
        hashed_password: Hashed password for authentication
        created_at: Timestamp when record was created
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"
