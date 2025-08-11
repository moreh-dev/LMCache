{{/*
Expand the name of the chart.
*/}}
{{- define "mooncake.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "mooncake.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" $name .Release.Name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mooncake.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mooncake.labels" -}}
{{- $ctx := index . 0 -}}
helm.sh/chart: {{ include "mooncake.chart" $ctx }}
{{ include "mooncake.selectorLabels" . }}
{{- if $ctx.Chart.AppVersion }}
app.kubernetes.io/version: {{ $ctx.Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ $ctx.Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mooncake.selectorLabels" -}}
{{- $ctx := index . 0 -}}
{{- $comp := index . 1 -}}
app.kubernetes.io/name: {{ include "mooncake.name" $ctx }}
app.kubernetes.io/instance: {{ $ctx.Release.Name }}
{{ include "mooncake.componentLabel" $comp }}
{{- end }}

{{- define "mooncake.componentLabel" -}}
app.kubernetes.io/component: {{ . | default "unknown" }}
{{- end }}