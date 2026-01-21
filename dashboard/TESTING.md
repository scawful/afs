# AFS Training Dashboard - Testing Guide

## Unit Tests

### API Endpoint Testing

```bash
# Start the server first
cd /Users/scawful/src/lab/afs/dashboard
python3 api.py &

# In another terminal, test endpoints
curl -s http://localhost:5000/api/health | jq .
curl -s http://localhost:5000/api/training/status | jq .
curl -s http://localhost:5000/api/models/status | jq .
curl -s http://localhost:5000/api/costs/breakdown | jq .
curl -s http://localhost:5000/api/metrics/gpu-utilization | jq .
curl -s http://localhost:5000/api/metrics/training-loss | jq .
curl -s http://localhost:5000/api/metrics/throughput | jq .
curl -s http://localhost:5000/api/models/registry | jq .
```

### API Response Validation

```bash
#!/bin/bash
# test_api.sh

API="http://localhost:5000"

echo "Testing API Endpoints..."
echo ""

# Test health
echo "1. Health Check"
curl -s $API/api/health | jq '.status' | grep -q "healthy" && echo "✓ PASS" || echo "✗ FAIL"

# Test training status
echo "2. Training Status"
curl -s $API/api/training/status | jq '.total_cost' | grep -q "[0-9]" && echo "✓ PASS" || echo "✗ FAIL"

# Test models status
echo "3. Models Status"
curl -s $API/api/models/status | jq '.models | length' | grep -q "[5-9]" && echo "✓ PASS" || echo "✗ FAIL"

# Test costs breakdown
echo "4. Costs Breakdown"
curl -s $API/api/costs/breakdown | jq '.total_cost' | grep -q "[0-9]" && echo "✓ PASS" || echo "✗ FAIL"

# Test metrics
echo "5. GPU Utilization"
curl -s $API/api/metrics/gpu-utilization | jq '.metrics | length' | grep -q "[0-9]" && echo "✓ PASS" || echo "✗ FAIL"

echo "6. Training Loss"
curl -s $API/api/metrics/training-loss | jq '.metrics | length' | grep -q "[0-9]" && echo "✓ PASS" || echo "✗ FAIL"

echo "7. Throughput"
curl -s $API/api/metrics/throughput | jq '.metrics | length' | grep -q "[0-9]" && echo "✓ PASS" || echo "✗ FAIL"

# Test registry
echo "8. Model Registry"
curl -s $API/api/models/registry | jq '.models | length' | grep -q "[5-9]" && echo "✓ PASS" || echo "✗ FAIL"

# Test export
echo "9. CSV Export"
curl -s $API/api/export/csv | head -1 | grep -q "Model Name" && echo "✓ PASS" || echo "✗ FAIL"

echo "10. JSON Export"
curl -s $API/api/export/json | jq '.models' > /dev/null && echo "✓ PASS" || echo "✗ FAIL"

echo ""
echo "All tests completed!"
