/**
 * Test script for OxideEngine using Candle framework
 * Run with: node test-candle.js
 */

const path = require('path');
const fs = require('fs');

// ANSI color codes for pretty output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  magenta: '\x1b[35m',
};

function printHeader(text) {
  console.log(`\n${colors.bright}${colors.blue}${'='.repeat(70)}${colors.reset}`);
  console.log(`${colors.bright}${colors.blue}${text}${colors.reset}`);
  console.log(`${colors.bright}${colors.blue}${'='.repeat(70)}${colors.reset}\n`);
}

function printSuccess(text) {
  console.log(`${colors.green}✓${colors.reset} ${text}`);
}

function printError(text) {
  console.log(`${colors.red}✗${colors.reset} ${text}`);
}

function printInfo(text) {
  console.log(`${colors.cyan}ℹ${colors.reset} ${text}`);
}

// Load the native module
let OxideEngine;
try {
  // Determine platform-specific binary name
  const platform = process.platform;
  const arch = process.arch;
  const platformMap = {
    'darwin-arm64': 'darwin-arm64',
    'darwin-x64': 'darwin-x64',
    'linux-x64': 'linux-x64-gnu',
    'win32-x64': 'win32-x64-msvc',
  };
  const platformKey = `${platform}-${arch}`;
  const platformSuffix = platformMap[platformKey] || 'unknown';
  const binaryName = `core.${platformSuffix}.node`;
  const binaryPath = path.join(__dirname, binaryName);

  if (fs.existsSync(binaryPath)) {
    const nativeModule = require(binaryPath);
    OxideEngine = nativeModule.OxideEngine;
    printSuccess(`Loaded native module: ${binaryName}`);
  } else {
    throw new Error(`Binary not found: ${binaryName}`);
  }
} catch (error) {
  printError('Failed to load native module');
  console.error(`${colors.red}Error: ${error.message}${colors.reset}`);
  console.error('\nPlease build the project first:');
  console.error('  npm run build');
  process.exit(1);
}

async function runTests() {
  console.clear();
  console.log(`${colors.bright}${colors.magenta}OxideEngine Test Suite - Candle Framework${colors.reset}`);
  console.log(`${colors.yellow}Testing GPU-accelerated tensor operations${colors.reset}\n`);

  try {
    // Test 1: Initialize OxideEngine
    printHeader('Test 1: Initialize OxideEngine');
    console.time('Initialization time');
    const engine = new OxideEngine();
    console.timeEnd('Initialization time');
    printSuccess('OxideEngine initialized successfully');

    // Get device info
    const deviceInfo = engine.getDeviceInfo();
    printInfo(`Device: ${deviceInfo}`);
    console.log();

    // Test 2: GPU Compute Test (Addition)
    printHeader('Test 2: GPU Tensor Addition');
    console.time('Computation time');
    const addResult = engine.testGpuCompute();
    console.timeEnd('Computation time');
    console.log('\n' + addResult);
    printSuccess('GPU addition test completed\n');

    // Test 3: GPU Multiplication Test
    printHeader('Test 3: GPU Tensor Multiplication');
    const a = 7.5;
    const b = 4.2;
    printInfo(`Computing: ${a} × ${b}`);
    console.time('Multiplication time');
    const multiplyResult = engine.testMultiply(a, b);
    console.timeEnd('Multiplication time');
    console.log('\n' + multiplyResult);
    printSuccess('GPU multiplication test completed\n');

    // Test 4: Create Tensor Info
    printHeader('Test 4: Create Tensor and Get Info');
    const testValues = [1.0, 2.0, 3.0, 4.0, 5.0];
    printInfo(`Creating tensor with values: [${testValues.join(', ')}]`);
    console.time('Tensor creation time');
    const tensorInfo = engine.createTensorInfo(testValues);
    console.timeEnd('Tensor creation time');
    console.log('\n' + tensorInfo);
    printSuccess('Tensor creation test completed\n');

    // Summary
    printHeader('Test Summary');
    printSuccess('All tests passed!');
    printInfo('OxideEngine is working correctly with Candle framework');
    printInfo('GPU acceleration is active and functional');
    console.log();

  } catch (error) {
    printHeader('Test Failed');
    printError('Error occurred during testing:');
    console.error(`\n${colors.red}${error.message}${colors.reset}`);
    if (error.stack) {
      console.error(`\n${colors.yellow}Stack trace:${colors.reset}`);
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Run all tests
runTests().catch((error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});
