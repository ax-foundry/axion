# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Loosened dependency version constraints for better compatibility
- Made search dependencies optional (`pip install axion[search]`)
- Made HuggingFace dependencies optional (`pip install axion[huggingface]`)
- Made Docling reader optional (`pip install axion[docling]`)

### Removed

- Removed unused `llama-index-readers-s3` dependency
