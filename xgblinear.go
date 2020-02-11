package leaves

// xgLinear is XGBoost model (gblinear)
type xgLinear struct {
	NumFeature       int
	nRawOutputGroups int
	BaseScore        float32
	Weights          []float32
}

func (e *xgLinear) NEstimators() int {
	return 1
}

func (e *xgLinear) NRawOutputGroups() int {
	return e.nRawOutputGroups
}

func (e *xgLinear) adjustNEstimators(nEstimators int) int {
	// gbliearn has only one estimator per class
	return 1
}

func (e *xgLinear) NFeatures() int {
	return e.NumFeature
}

func (e *xgLinear) Name() string {
	return "xgboost.gblinear"
}

func (e *xgLinear) predictInner(fvals []float32, nIterations int, predictions []float32, startIndex int) {
	for k := 0; k < e.nRawOutputGroups; k++ {
		predictions[startIndex+k] = e.BaseScore + float32(e.Weights[e.nRawOutputGroups*e.NumFeature+k])
		for i := 0; i < e.NumFeature; i++ {
			predictions[startIndex+k] += fvals[i] * float32(e.Weights[e.nRawOutputGroups*i+k])
		}
	}
}

func (e *xgLinear) resetFVals(fvals []float32) {
	for j := 0; j < len(fvals); j++ {
		fvals[j] = 0.0
	}
}
