#!/bin/bash

# RetentionAI Production Validation Script
# This script validates the production deployment by running comprehensive tests

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
TARGET_HOST="${1:-http://localhost}"
VALIDATION_TIMEOUT=300
TEST_DATA_FILE="validation_test_data.json"

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_header() { echo -e "${BOLD}${BLUE}=== $1 ===${NC}"; }

# Initialize validation results
VALIDATION_RESULTS=""
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

add_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    if [ "$status" == "PASS" ]; then
        PASSED_TESTS=$((PASSED_TESTS + 1))
        print_success "$test_name: $details"
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        print_error "$test_name: $details"
    fi
    
    VALIDATION_RESULTS+="\n$test_name: $status - $details"
}

# Test 1: Container Health
test_container_health() {
    print_header "Testing Container Health"
    
    # Check if containers are running
    containers=("retentionai-app" "retentionai-prometheus" "retentionai-grafana")
    
    for container in "${containers[@]}"; do
        if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
            add_result "Container $container" "PASS" "Running and healthy"
        else
            add_result "Container $container" "FAIL" "Not running or unhealthy"
        fi
    done
    
    # Check Docker Compose status
    if docker-compose -f docker-compose.production.yml ps | grep -q "Up"; then
        add_result "Docker Compose Stack" "PASS" "All services are up"
    else
        add_result "Docker Compose Stack" "FAIL" "Some services are down"
    fi
}

# Test 2: Endpoint Accessibility
test_endpoint_accessibility() {
    print_header "Testing Endpoint Accessibility"
    
    endpoints=(
        "$TARGET_HOST/:Main Application:200"
        "$TARGET_HOST/health:Health Check:200"
        "$TARGET_HOST:3000:Grafana:200"
        "$TARGET_HOST:9090:Prometheus:200"
        "$TARGET_HOST:9093:Alertmanager:200"
        "$TARGET_HOST:5000:MLflow:200"
    )
    
    for endpoint in "${endpoints[@]}"; do
        url=$(echo $endpoint | cut -d: -f1-2)
        name=$(echo $endpoint | cut -d: -f3)
        expected_code=$(echo $endpoint | cut -d: -f4)
        
        response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 10 "$url" || echo "000")
        
        if [ "$response_code" == "$expected_code" ]; then
            add_result "Endpoint $name" "PASS" "Accessible (HTTP $response_code)"
        else
            add_result "Endpoint $name" "FAIL" "Not accessible (HTTP $response_code)"
        fi
    done
}

# Test 3: Health Check Validation
test_health_checks() {
    print_header "Testing Health Check Endpoints"
    
    # Test general health
    health_response=$(curl -s "$TARGET_HOST/health" || echo "error")
    if echo "$health_response" | grep -q "overall_status"; then
        overall_status=$(echo "$health_response" | python3 -c "import sys, json; print(json.load(sys.stdin)['overall_status'])" 2>/dev/null || echo "unknown")
        if [ "$overall_status" == "healthy" ]; then
            add_result "General Health Check" "PASS" "System is healthy"
        else
            add_result "General Health Check" "FAIL" "System status: $overall_status"
        fi
    else
        add_result "General Health Check" "FAIL" "Health endpoint not responding correctly"
    fi
    
    # Test readiness
    readiness_code=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_HOST:8001/ready" || echo "000")
    if [ "$readiness_code" == "200" ]; then
        add_result "Readiness Check" "PASS" "Application is ready"
    else
        add_result "Readiness Check" "FAIL" "Readiness check failed (HTTP $readiness_code)"
    fi
    
    # Test liveness  
    liveness_code=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_HOST:8001/live" || echo "000")
    if [ "$liveness_code" == "200" ]; then
        add_result "Liveness Check" "PASS" "Application is alive"
    else
        add_result "Liveness Check" "FAIL" "Liveness check failed (HTTP $liveness_code)"
    fi
}

# Test 4: Metrics Collection
test_metrics_collection() {
    print_header "Testing Metrics Collection"
    
    # Test Prometheus metrics endpoint
    metrics_response=$(curl -s "$TARGET_HOST:8000/metrics" || echo "error")
    if echo "$metrics_response" | grep -q "retentionai_"; then
        metric_count=$(echo "$metrics_response" | grep "^retentionai_" | wc -l)
        add_result "Application Metrics" "PASS" "Found $metric_count custom metrics"
    else
        add_result "Application Metrics" "FAIL" "No custom metrics found"
    fi
    
    # Test Prometheus scraping
    prometheus_targets=$(curl -s "$TARGET_HOST:9090/api/v1/targets" || echo "error")
    if echo "$prometheus_targets" | grep -q "retentionai-app"; then
        add_result "Prometheus Scraping" "PASS" "Targets are being scraped"
    else
        add_result "Prometheus Scraping" "FAIL" "No application targets found"
    fi
    
    # Test Grafana data sources
    grafana_datasources=$(curl -s -u admin:retentionai123 "$TARGET_HOST:3000/api/datasources" || echo "error")
    if echo "$grafana_datasources" | grep -q "prometheus"; then
        add_result "Grafana Data Sources" "PASS" "Prometheus data source configured"
    else
        add_result "Grafana Data Sources" "FAIL" "Prometheus data source not found"
    fi
}

# Test 5: Database Connectivity
test_database_connectivity() {
    print_header "Testing Database Connectivity"
    
    # Test database through health endpoint
    db_health=$(curl -s "$TARGET_HOST:8001/health/database" || echo "error")
    if echo "$db_health" | grep -q "healthy"; then
        add_result "Database Connectivity" "PASS" "Database is accessible"
    else
        add_result "Database Connectivity" "FAIL" "Database connectivity issues"
    fi
    
    # Check if database file exists in container
    if docker exec retentionai-app test -f /app/data/retentionai.db; then
        add_result "Database File" "PASS" "Database file exists"
    else
        add_result "Database File" "FAIL" "Database file not found"
    fi
}

# Test 6: ML Model Availability
test_ml_model_availability() {
    print_header "Testing ML Model Availability"
    
    # Test model availability through health endpoint
    model_health=$(curl -s "$TARGET_HOST:8001/health/model_availability" || echo "error")
    if echo "$model_health" | grep -q "healthy"; then
        add_result "Model Availability" "PASS" "ML models are available"
    else
        add_result "Model Availability" "FAIL" "ML models not available"
    fi
    
    # Test MLflow connectivity
    mlflow_response=$(curl -s "$TARGET_HOST:5000/health" || echo "error")
    mlflow_code=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_HOST:5000" || echo "000")
    if [ "$mlflow_code" == "200" ]; then
        add_result "MLflow Server" "PASS" "MLflow tracking server is accessible"
    else
        add_result "MLflow Server" "FAIL" "MLflow tracking server not accessible"
    fi
}

# Test 7: Performance Validation
test_performance() {
    print_header "Testing Performance"
    
    # Test response time for health endpoint
    health_time=$(curl -s -w "%{time_total}" -o /dev/null "$TARGET_HOST/health")
    if (( $(echo "$health_time < 1.0" | bc -l) )); then
        add_result "Health Endpoint Performance" "PASS" "Response time: ${health_time}s"
    else
        add_result "Health Endpoint Performance" "FAIL" "Slow response time: ${health_time}s"
    fi
    
    # Test response time for main application
    app_time=$(curl -s -w "%{time_total}" -o /dev/null "$TARGET_HOST/" || echo "999")
    if (( $(echo "$app_time < 5.0" | bc -l) )); then
        add_result "Application Performance" "PASS" "Response time: ${app_time}s"
    else
        add_result "Application Performance" "FAIL" "Slow response time: ${app_time}s"
    fi
}

# Test 8: Security Validation
test_security() {
    print_header "Testing Security Configuration"
    
    # Test if sensitive endpoints are protected
    metrics_unauth=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_HOST/metrics" || echo "000")
    if [ "$metrics_unauth" == "403" ] || [ "$metrics_unauth" == "401" ]; then
        add_result "Metrics Security" "PASS" "Metrics endpoint is protected"
    else
        add_result "Metrics Security" "WARNING" "Metrics endpoint may be publicly accessible"
    fi
    
    # Test Grafana authentication
    grafana_unauth=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_HOST:3000/api/dashboards/home" || echo "000")
    if [ "$grafana_unauth" == "401" ] || [ "$grafana_unauth" == "403" ]; then
        add_result "Grafana Security" "PASS" "Grafana requires authentication"
    else
        add_result "Grafana Security" "WARNING" "Grafana may not require authentication"
    fi
}

# Test 9: Monitoring and Alerting
test_monitoring() {
    print_header "Testing Monitoring and Alerting"
    
    # Test Alertmanager
    alertmanager_status=$(curl -s "$TARGET_HOST:9093/api/v1/status" || echo "error")
    if echo "$alertmanager_status" | grep -q "success"; then
        add_result "Alertmanager Status" "PASS" "Alertmanager is running"
    else
        add_result "Alertmanager Status" "FAIL" "Alertmanager not responding"
    fi
    
    # Check if there are active alerts
    active_alerts=$(curl -s "$TARGET_HOST:9093/api/v1/alerts" || echo "error")
    if echo "$active_alerts" | grep -q "data"; then
        alert_count=$(echo "$active_alerts" | python3 -c "import sys, json; print(len(json.load(sys.stdin)['data']))" 2>/dev/null || echo "0")
        if [ "$alert_count" == "0" ]; then
            add_result "Active Alerts" "PASS" "No active alerts"
        else
            add_result "Active Alerts" "WARNING" "$alert_count active alerts"
        fi
    else
        add_result "Active Alerts" "FAIL" "Cannot retrieve alert status"
    fi
}

# Test 10: Data Pipeline Validation
test_data_pipeline() {
    print_header "Testing Data Pipeline"
    
    # Check if data directory exists and is accessible
    if docker exec retentionai-app test -d /app/data; then
        add_result "Data Directory" "PASS" "Data directory is accessible"
    else
        add_result "Data Directory" "FAIL" "Data directory not found"
    fi
    
    # Check for sample data file
    if docker exec retentionai-app test -f /app/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv; then
        add_result "Sample Data" "PASS" "Sample data file is present"
    else
        add_result "Sample Data" "WARNING" "Sample data file not found"
    fi
}

# Test 11: Backup and Recovery
test_backup_recovery() {
    print_header "Testing Backup and Recovery"
    
    # Check if backup script exists
    if docker exec retentionai-app test -f /opt/retentionai/scripts/backup.sh; then
        add_result "Backup Script" "PASS" "Backup script is available"
    else
        add_result "Backup Script" "WARNING" "Backup script not found"
    fi
    
    # Check if backup directory is accessible
    if [ -d "/opt/retentionai/backups" ]; then
        add_result "Backup Directory" "PASS" "Backup directory exists"
    else
        add_result "Backup Directory" "WARNING" "Backup directory not found"
    fi
}

# Test 12: Load Test Validation
test_load_handling() {
    print_header "Testing Load Handling Capability"
    
    # Simple concurrent request test
    print_status "Running basic load test..."
    
    # Test with 5 concurrent requests
    for i in {1..5}; do
        curl -s "$TARGET_HOST/health" > /dev/null &
    done
    wait
    
    # Check if service is still responsive
    post_load_response=$(curl -s -o /dev/null -w "%{http_code}" "$TARGET_HOST/health")
    if [ "$post_load_response" == "200" ]; then
        add_result "Load Handling" "PASS" "Service remains responsive under basic load"
    else
        add_result "Load Handling" "FAIL" "Service degraded under basic load"
    fi
}

# Generate validation report
generate_validation_report() {
    print_header "Generating Validation Report"
    
    report_file="validation_report_$(date +%Y%m%d_%H%M%S).html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>RetentionAI Production Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #1f77b4; color: white; padding: 20px; text-align: center; }
        .summary { background-color: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .pass { color: #28a745; font-weight: bold; }
        .fail { color: #dc3545; font-weight: bold; }
        .warning { color: #ffc107; font-weight: bold; }
        .test-result { margin: 10px 0; padding: 10px; border-left: 4px solid #ddd; }
        .test-pass { border-left-color: #28a745; }
        .test-fail { border-left-color: #dc3545; }
        .test-warning { border-left-color: #ffc107; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RetentionAI Production Validation Report</h1>
        <p>Validation completed on $(date)</p>
        <p>Target: $TARGET_HOST</p>
    </div>

    <div class="summary">
        <h2>Validation Summary</h2>
        <p><strong>Total Tests:</strong> $TOTAL_TESTS</p>
        <p><strong class="pass">Passed:</strong> $PASSED_TESTS</p>
        <p><strong class="fail">Failed:</strong> $FAILED_TESTS</p>
        <p><strong>Success Rate:</strong> $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%</p>
    </div>

    <div class="results">
        <h2>Detailed Results</h2>
EOF

    # Add test results to report
    echo -e "$VALIDATION_RESULTS" | while IFS= read -r line; do
        if [[ $line == *"PASS"* ]]; then
            echo "<div class='test-result test-pass'>$line</div>" >> "$report_file"
        elif [[ $line == *"FAIL"* ]]; then
            echo "<div class='test-result test-fail'>$line</div>" >> "$report_file"
        elif [[ $line == *"WARNING"* ]]; then
            echo "<div class='test-result test-warning'>$line</div>" >> "$report_file"
        fi
    done

    cat >> "$report_file" << EOF
    </div>

    <div class="summary">
        <h2>Recommendations</h2>
        <ul>
            <li>Address any failed tests before production go-live</li>
            <li>Monitor the application closely for the first 24 hours</li>
            <li>Set up automated health checks and alerts</li>
            <li>Schedule regular validation runs</li>
            <li>Review and update validation criteria as needed</li>
        </ul>
    </div>
</body>
</html>
EOF
    
    print_success "Validation report generated: $report_file"
}

# Main validation workflow
main() {
    print_header "Starting Production Validation"
    print_status "Target: $TARGET_HOST"
    print_status "Timeout: ${VALIDATION_TIMEOUT}s"
    
    # Run all validation tests
    test_container_health
    test_endpoint_accessibility
    test_health_checks
    test_metrics_collection
    test_database_connectivity
    test_ml_model_availability
    test_performance
    test_security
    test_monitoring
    test_data_pipeline
    test_backup_recovery
    test_load_handling
    
    # Generate report
    generate_validation_report
    
    # Print summary
    print_header "Validation Summary"
    print_status "Total tests: $TOTAL_TESTS"
    print_success "Passed: $PASSED_TESTS"
    print_error "Failed: $FAILED_TESTS"
    
    success_rate=$(( PASSED_TESTS * 100 / TOTAL_TESTS ))
    if [ $success_rate -ge 90 ]; then
        print_success "Validation passed with $success_rate% success rate"
        exit 0
    elif [ $success_rate -ge 70 ]; then
        print_warning "Validation completed with warnings ($success_rate% success rate)"
        exit 1
    else
        print_error "Validation failed with $success_rate% success rate"
        exit 2
    fi
}

# Run validation
main "$@"