/**
 * Test script for WebGPU Rust native module
 * Run with: node test.js
 */

const path = require('path');

// Try to load the native module from different possible locations
let nativeModule;
const fs = require('fs');

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

try {
  // Try loading NAPI-generated binary from root
  const binaryPath = path.join(__dirname, binaryName);
  if (fs.existsSync(binaryPath)) {
    nativeModule = require(binaryPath);
  } else {
    throw new Error(`Binary not found: ${binaryName}`);
  }
} catch (err) {
  try {
    // Try loading from index.node
    nativeModule = require('./index.node');
  } catch (err2) {
    console.error('❌ Failed to load native module. Please build the project first:');
    console.error('   npm run build');
    console.error(`\nLooking for: ${binaryName}`);
    console.error(`Error: ${err.message}`);
    process.exit(1);
  }
}

const { initializeWebgpu, listAdapters, getGpuCapabilities } = nativeModule;

// ANSI color codes for pretty output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  green: '\x1b[32m',
  blue: '\x1b[34m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
};

function printHeader(text) {
  console.log(`\n${colors.bright}${colors.blue}${'='.repeat(60)}${colors.reset}`);
  console.log(`${colors.bright}${colors.blue}${text}${colors.reset}`);
  console.log(`${colors.bright}${colors.blue}${'='.repeat(60)}${colors.reset}\n`);
}

function printSuccess(text) {
  console.log(`${colors.green}✓${colors.reset} ${text}`);
}

function printError(text) {
  console.log(`${colors.red}✗${colors.reset} ${text}`);
}

function printInfo(key, value) {
  console.log(`${colors.cyan}${key}:${colors.reset} ${value}`);
}

async function runTests() {
  console.log(`${colors.bright}WebGPU Native Module Test Suite${colors.reset}`);
  console.log(`${colors.yellow}Testing Rust-NAPI bindings for WebGPU${colors.reset}\n`);

  try {
    // Test 1: Initialize WebGPU
    printHeader('Test 1: Initialize WebGPU');
    console.time('Initialization time');
    const initMessage = await initializeWebgpu();
    console.timeEnd('Initialization time');
    console.log('\n' + initMessage);
    printSuccess('WebGPU initialized successfully\n');

    // Test 2: List all available adapters
    printHeader('Test 2: List Available GPU Adapters');
    console.time('List adapters time');
    const adapters = await listAdapters();
    console.timeEnd('List adapters time');

    if (adapters.length === 0) {
      printError('No GPU adapters found');
    } else {
      printSuccess(`Found ${adapters.length} GPU adapter(s):`);
      adapters.forEach((adapter, index) => {
        console.log(`  ${index + 1}. ${adapter}`);
      });
    }
    console.log();

    // Test 3: Get detailed GPU capabilities
    printHeader('Test 3: Get Detailed GPU Capabilities');
    console.time('Get capabilities time');
    const capabilities = await getGpuCapabilities();
    console.timeEnd('Get capabilities time');

    printSuccess('GPU Capabilities retrieved:\n');
    printInfo('Name', capabilities.name);
    printInfo('Backend', capabilities.backend);
    printInfo('Max Texture 2D', capabilities.maxTextureDimension2D + 'px');
    printInfo('Max Bind Groups', capabilities.maxBindGroups);
    printInfo('Max Buffer Size', formatBytes(capabilities.maxBufferSize));
    printInfo('Timestamp Queries', capabilities.supportsTimestamps ? 'Supported ✓' : 'Not supported ✗');

    // Summary
    printHeader('Test Summary');
    printSuccess('All tests passed!');
    printInfo('Status', 'WebGPU is ready to use');
    console.log();

  } catch (error) {
    printHeader('Test Failed');
    printError('Error occurred during testing:');
    console.error(`\n${colors.red}${error.message}${colors.reset}`);
    console.error(error.stack);
    process.exit(1);
  }
}

// Helper function to format bytes
function formatBytes(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
}

// Run all tests
console.clear();
runTests().catch((error) => {
  console.error('Unhandled error:', error);
  process.exit(1);
});
