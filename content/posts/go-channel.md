+++
title = "Simulating a go channel in Go"
date = "2023-06-28"
+++

The interface that we want to support ?

```go
type BuffChan struct {
    /* ... */
}

func NewBuffChan(size int) *BuffChan
func (c *BuffChan) Read() (int, bool)
func (c *BuffChan) Write(elem int)
func (c *BuffChan) Close()
```

Replicate the functionality of a go channel using primitives from the `sync`
package.