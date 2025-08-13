# Install LMCache with Helm Chart

### namespace
create namespace to run lmcache

```
kubectl create ns [namespace name]
```

## LMCache with CPU Offloading
create inference with lmcache at cpu offloading

```
helm install [lmcache name] ./lmcache -f values/lmcache-distilgpt2-cpu-offloading-values.yaml -n [namespace]
```

## LMCache with Centralized Sharing [ w. Infinistore ]
create inference with infinistore lmcache 
- decide link type IB/Ethernet for infinistore
  - change `--link-type` at `infinistore.Value.arg`
  - change `infinistore_link_type` at `lmcache.Value.lmcache.config`
- change `remote_url` at `lmcache.Value.lmcache.config`
  - `remote_url: infinistore://[pod name].[service name]:[service port]?device=[device name]`
    - `service port` must be included

```
# start infinistore lmcache
helm install [infinistore name] ./infinistore -f values/infinistore-values.yaml -n [namespace]

# start inference with lmcache
helm install [lmcache name] ./lmcache -f values/lmcache-distilgpt2-centralized-sharing-infinistore-values.yaml -n [namespace]
```

## LMCache with P2P Sharing [ w. mooncake ]
create inference with mooncake lmcache
- change urls at `lmcache.Value.lmcache.config`
  - `remote_url: mooncakestore://[master service url]:[master service port]/`
    - `service port` must be included
  - `metadata_server: "http://[meta service url]"`
  - `master_server_address: "[master service url]"`

```
# start mooncake meta & master
helm install [mooncake name] ./mooncake -f values/mooncake-values.yaml -n [namespace]

# start inference with lmcache
# fix mooncake url
helm install [lmcache name] ./lmcache -f values/lmcache-distilgpt2-p2p-sharing-mooncake-values.yaml -n [namespace]
```
 