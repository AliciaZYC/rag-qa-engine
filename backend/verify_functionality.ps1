Write-Host "=== VERIFICATION CHECKLIST ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Frontend accessible:" -NoNewline
try { 
    $response = Invoke-WebRequest -Uri "http://localhost:5173" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    Write-Host "   ✓ YES" -ForegroundColor Green
} catch { 
    Write-Host "   ✗ NO" -ForegroundColor Red 
}

Write-Host "2. Backend healthy:" -NoNewline
try { 
    $response = Invoke-RestMethod -Uri "http://localhost:5001/api/health" -TimeoutSec 5 -ErrorAction Stop
    if ($response.status -eq "healthy") {
        Write-Host "   ✓ YES" -ForegroundColor Green
    } else {
        Write-Host "   ✗ NO" -ForegroundColor Red
    }
} catch { 
    Write-Host "   ✗ NO" -ForegroundColor Red 
}

Write-Host "3. Database has 100 docs:" -NoNewline
try {
    $count = docker-compose exec -T postgres psql -U rag_user -d rag_db -t -c "SELECT COUNT(*) FROM documents;" 2>$null
    if ($count -match "100") {
        Write-Host "   ✓ YES" -ForegroundColor Green
    } else {
        Write-Host "   ✗ NO (found: $($count.Trim()))" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ NO" -ForegroundColor Red
}

Write-Host "4. New columns exist:" -NoNewline
try {
    $schema = docker-compose exec -T postgres psql -U rag_user -d rag_db -c "\d documents" 2>$null
    if ($schema -match "provision_label") {
        Write-Host "   ✓ YES" -ForegroundColor Green
    } else {
        Write-Host "   ✗ NO" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ NO" -ForegroundColor Red
}

Write-Host "5. LegalBERT embeddings (768-dim):" -NoNewline
try {
    $dim = docker-compose exec -T backend python -c "from embeddings import DualEmbedder, ModelType; e = DualEmbedder(primary_model=ModelType.LEGAL_BERT, use_fallback=False); print(e.get_dimension())" 2>$null
    if ($dim -match "768") {
        Write-Host "   ✓ YES" -ForegroundColor Green
    } else {
        Write-Host "   ✗ NO (found: $($dim.Trim()))" -ForegroundColor Red
    }
} catch {
    Write-Host "   ✗ NO" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== END OF VERIFICATION ===" -ForegroundColor Cyan