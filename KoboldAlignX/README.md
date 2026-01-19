# KoboldAlignX

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-Proprietary-red.svg)

A robust FastAPI + Celery application for processing job data from Reliance MRL Solutions and delivering formatted Excel reports via email.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Deployment](#deployment)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## ✨ Features

- **Automated Job Processing**: Periodic polling and processing of jobs from Reliance API
- **OAuth 2.0 Authentication**: Secure authentication with external APIs
- **Excel Report Generation**: Automated creation of formatted Excel reports
- **Email Notifications**: HTML email delivery with attachments via Office 365
- **Async Task Processing**: Celery-based distributed task queue
- **RESTful API**: FastAPI endpoints for job management
- **Error Handling**: Comprehensive error handling and retry logic
- **Type Safety**: Full type hints and Pydantic validation
- **Docker Support**: Containerized deployment with Docker Compose

## 🏗️ Architecture

```
┌──────────────┐
│   FastAPI    │  ← HTTP REST API
│  Application │
└──────┬───────┘
       │
       ├─→ RabbitMQ ─→ Celery Workers ─→ Redis
       │      ↓                ↓
       │   [Tasks]        [Results]
       │      ↓                ↓
       └─→ PostgreSQL ←──── Storage Service
              ↓
       ┌────────────────┐
       │  External APIs │
       ├────────────────┤
       │ • Reliance API │
       │ • Kobold API   │
       │ • SMTP (O365)  │
       └────────────────┘
```

### Project Structure

```
src/
├── api/                    # FastAPI routes & endpoints
│   ├── health.py          # Health check endpoints
│   └── jobs.py            # Job processing endpoints
├── core/
│   ├── config.py          # Configuration management
│   ├── exceptions.py      # Custom exception classes
│   └── dependencies.py    # FastAPI dependencies
├── services/              # Business logic layer
│   ├── reliance_api.py   # Reliance API client
│   ├── job_processor.py  # Job processing logic
│   ├── email_service.py  # Email functionality
│   └── storage_service.py # Data persistence
├── tasks/                 # Celery tasks
│   ├── celery_app.py     # Celery configuration
│   └── job_tasks.py      # Job processing tasks
├── models/                # Data models
│   ├── database.py       # SQLAlchemy models
│   └── schemas.py        # Pydantic schemas
├── utils/                 # Utility functions
│   ├── location_parser.py
│   ├── excel_generator.py
│   └── validators.py
└── main.py               # Application entry point
```

## 📦 Prerequisites

- **Python**: 3.9 or higher
- **Docker**: 20.10 or higher (for containerized deployment)
- **Docker Compose**: 2.0 or higher
- **PostgreSQL**: 13 or higher
- **Redis**: 6.0 or higher
- **RabbitMQ**: 3.9 or higher

## 🚀 Installation

### Local Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd KoboldAlignX
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
   # For development
   pip install -r requirements-dev.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual configuration values
   ```

5. **Initialize database**
   ```bash
   alembic upgrade head
   ```

### Docker Setup

1. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Build and start services**
   ```bash
   docker-compose -f docker-compose-prod.yml build
   docker-compose -f docker-compose-prod.yml up -d
   ```

3. **Verify services are running**
   ```bash
   docker-compose ps
   curl http://localhost/health
   ```

## ⚙️ Configuration

### Required Environment Variables

All configuration is managed through environment variables. See `.env.example` for a complete list.

**Critical variables that MUST be set:**

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Reliance API
RELIANCE_CLIENT_ID=your_client_id
RELIANCE_CLIENT_SECRET=your_client_secret

# Email
EMAIL_USERNAME=your_email@domain.com
EMAIL_PASSWORD=your_password

# Security
SECRET_KEY=your_secret_key_here
```

### Generating Secret Key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 💻 Usage

### Starting the Application

**Local Development:**
```bash
# Start API server
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker (in another terminal)
celery -A src.tasks.celery_app worker --loglevel=info

# Start Celery beat scheduler (in another terminal)
celery -A src.tasks.celery_app beat --loglevel=info
```

**Docker:**
```bash
docker-compose -f docker-compose-prod.yml up
```

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Create Job (Synchronous)
```bash
POST /reliancejob/{job_id}
Content-Type: application/json

{
  "amount": 100,
  "x": "example_x",
  "y": "example_y"
}
```

#### Create Job (Asynchronous)
```bash
POST /reliancejob/{job_id}/async
Content-Type: application/json

{
  "amount": 100,
  "x": "example_x",
  "y": "example_y"
}
```

#### Get Task Status
```bash
GET /task/{task_id}
```

## 📚 API Documentation

Once the application is running, interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🛠️ Development

### Code Quality Tools

**Format code:**
```bash
black src/
isort src/
```

**Lint code:**
```bash
ruff check src/
mypy src/
```

**Run security checks:**
```bash
bandit -r src/
```

### Pre-commit Hooks

Install pre-commit hooks to automatically check code quality:

```bash
pre-commit install
pre-commit run --all-files
```

### Database Migrations

**Create a new migration:**
```bash
alembic revision --autogenerate -m "Description of changes"
```

**Apply migrations:**
```bash
alembic upgrade head
```

**Rollback migration:**
```bash
alembic downgrade -1
```

## 🚢 Deployment

### Production Deployment

1. **Build Docker images**
   ```bash
   docker-compose -f docker-compose-prod.yml build
   ```

2. **Push to container registry**
   ```bash
   docker tag koboldalignx-app:latest your-registry.azurecr.io/koboldalignx-app:v2.0.0
   docker push your-registry.azurecr.io/koboldalignx-app:v2.0.0
   ```

3. **Deploy to production**
   ```bash
   # Use your deployment method (Kubernetes, Azure Container Instances, etc.)
   kubectl apply -f k8s/deployment.yaml
   ```

### Environment-Specific Configurations

- **Development**: `docker-compose-dev.yml`
- **Production**: `docker-compose-prod.yml`

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_email_service.py

# Run with verbose output
pytest -v
```

### Test Coverage

View coverage report:
```bash
open htmlcov/index.html  # On macOS
# Or navigate to htmlcov/index.html in your browser
```

## 🔧 Troubleshooting

### Common Issues

**Issue: Database connection failed**
```
Solution: Verify DATABASE_URL in .env and ensure PostgreSQL is running
```

**Issue: Celery worker not processing tasks**
```
Solution: Check RabbitMQ connection and ensure worker is started
```

**Issue: Email sending fails**
```
Solution: Verify Office 365 credentials and SMTP settings
```

### Logs

**View application logs:**
```bash
# Docker
docker-compose logs -f alignxapp

# Local
tail -f logs/app.log
```

**View Celery worker logs:**
```bash
# Docker
docker-compose logs -f celery_worker

# Local
tail -f logs/celery.log
```

## 🤝 Contributing

### Development Workflow

1. Create a feature branch
2. Make your changes
3. Run tests and linting
4. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all functions
- Write docstrings for all public methods
- Keep functions focused and small (<50 lines)
- Use meaningful variable names

## 📝 License

Proprietary - Kobold Completions Inc.

## 📞 Support

For support, contact: support@koboldinc.com

---

## 🔒 Security

### Security Best Practices

1. **Never commit credentials** to version control
2. **Use environment variables** for all secrets
3. **Rotate credentials** regularly
4. **Enable HTTPS** in production
5. **Keep dependencies** up to date

### Reporting Security Issues

Email security concerns to: security@koboldinc.com

---

**Last Updated**: January 2025
**Version**: 2.0.0
