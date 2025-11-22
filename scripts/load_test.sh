#!/bin/bash

# RetentionAI Load Testing Script
# This script performs comprehensive load testing of the production deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
TARGET_HOST="${1:-http://localhost}"
CONCURRENT_USERS="${2:-10}"
TEST_DURATION="${3:-60}"
RAMP_UP_TIME="${4:-10}"

print_header() { echo -e "${BLUE}=== $1 ===${NC}"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    if ! command -v curl &> /dev/null; then
        print_error "curl is required but not installed"
        exit 1
    fi
    
    if ! command -v ab &> /dev/null; then
        print_warning "Apache Bench (ab) not found. Installing..."
        # Try to install ab
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y apache2-utils
        elif command -v yum &> /dev/null; then
            sudo yum install -y httpd-tools
        elif command -v brew &> /dev/null; then
            brew install httpd
        else
            print_error "Cannot install Apache Bench. Please install manually."
            exit 1
        fi
    fi
    
    print_success "All dependencies are available"
}

# Test basic connectivity
test_connectivity() {
    print_header "Testing Basic Connectivity"
    
    endpoints=(
        "${TARGET_HOST}/:Main Application"
        "${TARGET_HOST}/health:Health Check"
        "${TARGET_HOST}:3000:Grafana"
        "${TARGET_HOST}:9090:Prometheus"
    )
    
    for endpoint in "${endpoints[@]}"; do
        url=$(echo $endpoint | cut -d: -f1-2)
        name=$(echo $endpoint | cut -d: -f3-)
        
        if curl -s -f --connect-timeout 5 "$url" > /dev/null; then
            print_success "$name is accessible"
        else
            print_error "$name is not accessible at $url"
        fi
    done
}

# Performance baseline test
run_baseline_test() {
    print_header "Running Baseline Performance Test"
    
    print_status "Testing health endpoint..."
    ab -n 100 -c 5 "${TARGET_HOST}/health" > baseline_health.txt 2>&1
    
    health_rps=$(grep "Requests per second" baseline_health.txt | awk '{print $4}')
    health_time=$(grep "Time per request.*mean" baseline_health.txt | head -1 | awk '{print $4}')
    
    print_success "Health endpoint: ${health_rps} req/sec, ${health_time}ms avg response time"
    
    print_status "Testing main application..."
    ab -n 50 -c 2 "${TARGET_HOST}/" > baseline_app.txt 2>&1
    
    app_rps=$(grep "Requests per second" baseline_app.txt | awk '{print $4}' || echo "N/A")
    app_time=$(grep "Time per request.*mean" baseline_app.txt | head -1 | awk '{print $4}' || echo "N/A")
    
    print_success "Main app: ${app_rps} req/sec, ${app_time}ms avg response time"
}

# Load test simulation
run_load_test() {
    print_header "Running Load Test"
    print_status "Concurrent Users: $CONCURRENT_USERS"
    print_status "Test Duration: ${TEST_DURATION}s"
    print_status "Target: $TARGET_HOST"
    
    # Create test data for prediction endpoint (if available)
    create_test_data() {
        cat > test_customer.json << EOF
{
    "customerID": "TEST001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "One year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Mailed check",
    "MonthlyCharges": 65.75,
    "TotalCharges": 1578.5
}
EOF
    }
    
    create_test_data
    
    # Test different endpoints with different load patterns
    print_status "Testing health endpoint under load..."
    ab -n $((CONCURRENT_USERS * 20)) -c $CONCURRENT_USERS -t $TEST_DURATION "${TARGET_HOST}/health" > load_health.txt 2>&1 &
    HEALTH_PID=$!
    
    print_status "Testing main application under load..."
    ab -n $((CONCURRENT_USERS * 5)) -c $((CONCURRENT_USERS / 2)) -t $TEST_DURATION "${TARGET_HOST}/" > load_app.txt 2>&1 &
    APP_PID=$!
    
    # Monitor system during load test
    print_status "Monitoring system metrics during load test..."
    monitor_system() {
        echo "timestamp,cpu_usage,memory_usage,load_avg" > system_metrics.csv
        for ((i=1; i<=TEST_DURATION; i++)); do
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 || echo "0")
            memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
            load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
            echo "$timestamp,$cpu_usage,$memory_usage,$load_avg" >> system_metrics.csv
            sleep 1
        done
    }
    monitor_system &
    MONITOR_PID=$!
    
    # Wait for tests to complete
    wait $HEALTH_PID
    wait $APP_PID
    kill $MONITOR_PID 2>/dev/null || true
    
    print_success "Load test completed"
}

# Stress test
run_stress_test() {
    print_header "Running Stress Test"
    
    stress_users=$((CONCURRENT_USERS * 3))
    stress_duration=30
    
    print_status "Stress testing with $stress_users concurrent users for ${stress_duration}s"
    
    ab -n $((stress_users * 10)) -c $stress_users -t $stress_duration "${TARGET_HOST}/health" > stress_test.txt 2>&1
    
    if [ $? -eq 0 ]; then
        print_success "Stress test completed successfully"
    else
        print_warning "Stress test completed with some failures"
    fi
}

# Memory leak test
run_memory_leak_test() {
    print_header "Running Memory Leak Test"
    
    print_status "Running extended test to check for memory leaks..."
    
    # Record initial memory
    initial_memory=$(docker stats --no-stream --format "{{.MemUsage}}" retentionai-app 2>/dev/null | cut -d'/' -f1 | sed 's/[^0-9.]//g' || echo "0")
    
    # Run sustained load for 5 minutes
    ab -n 1000 -c 5 -t 300 "${TARGET_HOST}/health" > memory_leak_test.txt 2>&1 &
    
    # Monitor memory every 30 seconds
    echo "time,memory_mb" > memory_usage.csv
    for i in {1..10}; do
        sleep 30
        memory=$(docker stats --no-stream --format "{{.MemUsage}}" retentionai-app 2>/dev/null | cut -d'/' -f1 | sed 's/[^0-9.]//g' || echo "0")
        echo "${i},$memory" >> memory_usage.csv
        print_status "Memory usage at ${i}0s: ${memory}MB"
    done
    
    # Check for memory growth
    final_memory=$(tail -1 memory_usage.csv | cut -d',' -f2)
    memory_growth=$(echo "$final_memory - $initial_memory" | bc 2>/dev/null || echo "0")
    
    if (( $(echo "$memory_growth > 100" | bc -l) )); then
        print_warning "Potential memory leak detected: ${memory_growth}MB growth"
    else
        print_success "No significant memory leak detected: ${memory_growth}MB growth"
    fi
}

# Generate load test report
generate_report() {
    print_header "Generating Load Test Report"
    
    report_file="load_test_report_$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>RetentionAI Load Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #1f77b4; color: white; padding: 20px; text-align: center; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { margin: 10px 0; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .error { color: #dc3545; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RetentionAI Load Test Report</h1>
        <p>Generated on $(date)</p>
    </div>

    <div class="section">
        <h2>Test Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Target Host</td><td>$TARGET_HOST</td></tr>
            <tr><td>Concurrent Users</td><td>$CONCURRENT_USERS</td></tr>
            <tr><td>Test Duration</td><td>${TEST_DURATION}s</td></tr>
            <tr><td>Ramp-up Time</td><td>${RAMP_UP_TIME}s</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>Performance Results</h2>
EOF
    
    # Add baseline results
    if [ -f baseline_health.txt ]; then
        echo "<h3>Baseline Performance</h3>" >> "$report_file"
        echo "<h4>Health Endpoint</h4>" >> "$report_file"
        grep -E "(Requests per second|Time per request|Transfer rate)" baseline_health.txt | sed 's/^/<p>/' | sed 's/$/<\/p>/' >> "$report_file"
        
        if [ -f baseline_app.txt ]; then
            echo "<h4>Main Application</h4>" >> "$report_file"
            grep -E "(Requests per second|Time per request|Transfer rate)" baseline_app.txt | sed 's/^/<p>/' | sed 's/$/<\/p>/' >> "$report_file"
        fi
    fi
    
    # Add load test results
    if [ -f load_health.txt ]; then
        echo "<h3>Load Test Results</h3>" >> "$report_file"
        echo "<h4>Health Endpoint Under Load</h4>" >> "$report_file"
        grep -E "(Requests per second|Time per request|Transfer rate|Failed requests)" load_health.txt | sed 's/^/<p>/' | sed 's/$/<\/p>/' >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF
    </div>

    <div class="section">
        <h2>System Metrics</h2>
EOF
    
    if [ -f system_metrics.csv ]; then
        echo "<p>System metrics were collected during the test. See system_metrics.csv for detailed data.</p>" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF
    </div>

    <div class="section">
        <h2>Test Files Generated</h2>
        <ul>
            <li>baseline_health.txt - Baseline health endpoint test</li>
            <li>baseline_app.txt - Baseline application test</li>
            <li>load_health.txt - Load test on health endpoint</li>
            <li>load_app.txt - Load test on main application</li>
            <li>system_metrics.csv - System metrics during load test</li>
            <li>memory_usage.csv - Memory usage over time</li>
        </ul>
    </div>

    <div class="section">
        <h2>Recommendations</h2>
        <ul>
            <li>Monitor the application for at least 24 hours after deployment</li>
            <li>Set up automated performance tests in CI/CD pipeline</li>
            <li>Configure alerting based on the observed baseline performance</li>
            <li>Consider horizontal scaling if response times exceed acceptable limits</li>
            <li>Review memory usage patterns for potential optimizations</li>
        </ul>
    </div>
</body>
</html>
EOF
    
    print_success "Report generated: $report_file"
}

# Cleanup test files
cleanup() {
    print_header "Cleaning Up Test Files"
    
    test_files=(
        "test_customer.json"
        "baseline_health.txt"
        "baseline_app.txt"
        "load_health.txt"
        "load_app.txt"
        "stress_test.txt"
        "memory_leak_test.txt"
        "system_metrics.csv"
        "memory_usage.csv"
    )
    
    for file in "${test_files[@]}"; do
        if [ -f "$file" ]; then
            rm "$file"
            print_status "Removed $file"
        fi
    done
}

# Main execution
main() {
    case "${1:-full}" in
        "quick")
            print_header "Quick Load Test"
            check_dependencies
            test_connectivity
            run_baseline_test
            ;;
        "full")
            print_header "Comprehensive Load Test"
            check_dependencies
            test_connectivity
            run_baseline_test
            run_load_test
            run_stress_test
            generate_report
            ;;
        "stress")
            print_header "Stress Test Only"
            check_dependencies
            test_connectivity
            run_stress_test
            ;;
        "memory")
            print_header "Memory Leak Test"
            check_dependencies
            test_connectivity
            run_memory_leak_test
            ;;
        "cleanup")
            cleanup
            ;;
        *)
            echo "RetentionAI Load Testing Script"
            echo ""
            echo "Usage: $0 [test_type] [host] [concurrent_users] [duration]"
            echo ""
            echo "Test Types:"
            echo "  quick    - Basic connectivity and baseline performance test"
            echo "  full     - Comprehensive load testing with report generation"  
            echo "  stress   - High-load stress testing"
            echo "  memory   - Extended test to check for memory leaks"
            echo "  cleanup  - Remove test files"
            echo ""
            echo "Examples:"
            echo "  $0 quick                           # Quick test with defaults"
            echo "  $0 full http://localhost 20 120    # Full test, 20 users, 120 seconds"
            echo "  $0 stress                          # Stress test"
            echo "  $0 memory                          # Memory leak test"
            exit 1
            ;;
    esac
}

# Trap for cleanup on interrupt
trap cleanup INT

# Run with all arguments
main "$@"