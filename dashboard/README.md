# Homeostasis Dashboard

Web dashboard for monitoring and managing Homeostasis self-healing activities.

## Features

- **Real-time monitoring** of errors and fixes
- **Historical data visualization** with charts and metrics
- **Fix approval interface** for reviewing and approving fixes
- **Canary deployment management** for gradual rollout of fixes
- **System configuration** through a user-friendly interface
- **Responsive design** that works on desktop and mobile devices

## Architecture

The dashboard consists of:

1. **Backend API**: Flask-based REST API to interact with Homeostasis
2. **Frontend UI**: HTML/CSS/JavaScript interface using Bootstrap and Chart.js
3. **WebSocket Service**: Real-time updates for monitoring

## Setup

### Prerequisites

- Python 3.8+
- Node.js 14+ (for frontend development tools - optional)
- Homeostasis core system

### Installation

```bash
# Install required packages
pip install -r dashboard/requirements.txt

# Start the dashboard
python dashboard/app.py
```

## Development

For development mode with auto-reload:

```bash
# Install development dependencies
pip install -r dashboard/requirements-dev.txt

# Run in development mode
python dashboard/app.py --debug
```

## Configuration

Edit `dashboard/config.yaml` to configure:

- Server host and port
- Authentication settings
- API endpoints
- Dashboard theme and refresh interval

## API Reference

The dashboard provides the following API endpoints:

- `/api/status`: Get dashboard status
- `/api/errors`: Get error information
- `/api/fixes`: Get fix information
- `/api/approvals`: Manage fix approvals
- `/api/metrics`: Get system metrics
- `/api/canary`: Manage canary deployments

## Authentication

The dashboard supports user authentication with the following default roles:

- **Admin**: Full access to all features
- **Operator**: Can approve/reject fixes and manage deployments
- **Viewer**: Read-only access to monitoring data

## Screenshots

(Screenshots will be added as the dashboard is developed)

## License

This project is part of the Homeostasis open-source framework and is available under the same license.