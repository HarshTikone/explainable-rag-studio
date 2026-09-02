# Ember Cache — Current Expiration Policy

Status: current. Ember caches retrieval results for 12 minutes and embedding results for 24 hours. Cache code CA-8801 means the namespace version is stale. Deployments increment the namespace before serving traffic, and Search Reliability owns invalidation.
