package ai

// Catalog is the definitions of the Runtime's app: Store.Catalog of its store, hwnpud's
// through it.
func (rt *Runtime) Catalog() (*Catalog, error) { return rt.app.Store.Catalog() }
