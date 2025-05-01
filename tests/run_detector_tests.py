#!/usr/bin/env python3
"""
Test runner for document detector module tests.
Provides a convenient way to run all tests or specific test suites.
"""

import unittest
import argparse
import sys
import os
import logging
from datetime import datetime
from unittest.mock import MagicMock

# Mock the magic module before importing any modules that depend on it
sys.modules['magic'] = MagicMock()

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import test modules
from tests.unit.test_document_detector import TestFileExtensionDetector, TestMagicNumberDetector, TestContentDetector, TestDocumentDetector, TestDetectionResult
from tests.unit.test_enhanced_detector_service import TestEnhancedDetectorService
from tests.unit.test_detector_factory import TestDetectorFactory
from tests.integration.test_detector_directory_loader import TestDetectorDirectoryLoaderIntegration, TestDetectorDirectoryLoaderWithRealFiles
from tests.integration.test_detector_pipeline_integration import TestDetectorPipelineIntegration, TestEndToEndPipelineIntegration


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'detector_tests_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def run_unit_tests(verbosity=2):
    """Run all unit tests for the document detector module."""
    print("\n=== Running Document Detector Unit Tests ===\n")
    
    # Create a test suite with all unit tests
    unit_suite = unittest.TestSuite()
    
    # Add document detector tests
    unit_suite.addTest(unittest.makeSuite(TestFileExtensionDetector))
    unit_suite.addTest(unittest.makeSuite(TestMagicNumberDetector))
    unit_suite.addTest(unittest.makeSuite(TestContentDetector))
    unit_suite.addTest(unittest.makeSuite(TestDocumentDetector))
    unit_suite.addTest(unittest.makeSuite(TestDetectionResult))
    
    # Add enhanced detector service tests
    unit_suite.addTest(unittest.makeSuite(TestEnhancedDetectorService))
    
    # Add detector factory tests
    unit_suite.addTest(unittest.makeSuite(TestDetectorFactory))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(unit_suite)


def run_integration_tests(verbosity=2, include_real_files=False):
    """Run all integration tests for the document detector module."""
    print("\n=== Running Document Detector Integration Tests ===\n")
    
    # Create a test suite with all integration tests
    integration_suite = unittest.TestSuite()
    
    # Add directory loader integration tests
    integration_suite.addTest(unittest.makeSuite(TestDetectorDirectoryLoaderIntegration))
    
    # Add pipeline integration tests
    integration_suite.addTest(unittest.makeSuite(TestDetectorPipelineIntegration))
    
    # Optionally add tests that use real files
    if include_real_files:
        print("\n=== Including Tests with Real Files ===\n")
        integration_suite.addTest(unittest.makeSuite(TestDetectorDirectoryLoaderWithRealFiles))
        integration_suite.addTest(unittest.makeSuite(TestEndToEndPipelineIntegration))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(integration_suite)


def run_all_tests(verbosity=2, include_real_files=False):
    """Run all tests for the document detector module."""
    print("\n=== Running All Document Detector Tests ===\n")
    
    # Run unit tests
    unit_result = run_unit_tests(verbosity)
    
    # Run integration tests
    integration_result = run_integration_tests(verbosity, include_real_files)
    
    # Return combined result
    return unit_result.wasSuccessful() and integration_result.wasSuccessful()


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description='Run document detector tests')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--all', action='store_true', help='Run all tests (default)')
    parser.add_argument('--real-files', action='store_true', help='Include tests with real files')
    parser.add_argument('--verbose', '-v', action='count', default=1, help='Increase verbosity')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    # Determine verbosity level
    verbosity = 0 if args.quiet else args.verbose + 1
    
    # Run the requested tests
    if args.unit:
        success = run_unit_tests(verbosity).wasSuccessful()
    elif args.integration:
        success = run_integration_tests(verbosity, args.real_files).wasSuccessful()
    else:  # Default to all tests
        success = run_all_tests(verbosity, args.real_files)
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
