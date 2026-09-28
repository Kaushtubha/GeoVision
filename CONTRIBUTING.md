# 🤝 Contributing to GeoVision

Thank you for your interest in contributing to GeoVision! We welcome contributions across all areas: deep learning architectures, geospatial tooling, FastAPI endpoints, React dashboard components, and documentation.

---

## 1. Development Workflow

1. **Fork & Clone**:
   ```bash
   git clone https://github.com/your-username/GeoVision.git
   cd GeoVision
   ```

2. **Setup Python Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[all]"
   ```

3. **Setup Frontend Environment**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

---

## 2. Code Style & Quality Standards

- **Python**: We format and lint with `ruff`.
  ```bash
  ruff check geovision tests
  ruff format geovision tests
  ```
- **Type Checking**: All public functions and Pydantic schemas must have complete type annotations.
- **Frontend**: Clean TypeScript with strict typing. Format with Prettier / ESLint.

---

## 3. Conventional Commit Messages

GeoVision follows the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat(...)`: A new feature or subsystem capability
- `fix(...)`: A bug fix
- `docs(...)`: Documentation changes
- `test(...)`: Adding or updating test suites
- `refactor(...)`: Code change that neither fixes a bug nor adds a feature
- `perf(...)`: Performance optimization
- `config(...)`: Tooling, Docker, or build configuration

---

## 4. Submitting a Pull Request

- Ensure all test suites pass: `pytest tests/`
- Add unit tests for any new algorithms or API endpoints.
- Keep PRs focused on a single responsibility.
