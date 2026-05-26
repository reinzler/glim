#!/usr/bin/env bash
set -euo pipefail

OWNER_REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

create_label() {
  local name="$1"
  local color="$2"
  local desc="$3"

  if gh label list --limit 500 | awk -F'\t' '{print $1}' | grep -Fxq "$name"; then
    echo "[OK] label exists: $name"
  else
    gh label create "$name" --color "$color" --description "$desc"
  fi
}

create_milestone() {
  local title="$1"
  local desc="$2"

  local existing
  existing=$(gh api "repos/${OWNER_REPO}/milestones?state=all" \
    --jq ".[] | select(.title == \"${title}\") | .number" || true)

  if [ -n "$existing" ]; then
    echo "[OK] milestone exists: $title (#$existing)"
  else
    gh api -X POST "repos/${OWNER_REPO}/milestones" \
      -f title="$title" \
      -f description="$desc" \
      -f state="open" >/dev/null
    echo "[NEW] milestone created: $title"
  fi
}

create_issue_if_missing() {
  local title="$1"
  local milestone="$2"
  local labels="$3"
  local body="$4"

  local existing
  existing=$(gh issue list --state all --search "$title in:title" --json number,title \
    --jq ".[] | select(.title == \"${title}\") | .number" || true)

  if [ -n "$existing" ]; then
    echo "[OK] issue exists: #$existing $title"
  else
    gh issue create \
      --title "$title" \
      --body "$body" \
      --milestone "$milestone" \
      --label "$labels" >/dev/null
    echo "[NEW] issue created: $title"
  fi
}

create_label "area:core" "1d76db" "Core GLIM changes"
create_label "area:preprocess" "0e8a16" "Point cloud preprocessing"
create_label "area:export" "5319e7" "Map and dataset export"
create_label "area:diagnostics" "fbca04" "Diagnostics and QualitySnapshot"
create_label "area:attributes" "c2e0c6" "PR290-style point attributes"
create_label "area:mapcleaner" "bfdadc" "MapCleaner-compatible integration"
create_label "type:milestone-task" "7057ff" "Explicit task for milestone tracking"
create_label "priority:p0" "b60205" "Must have for current milestone"

create_milestone \
  "M0: Safe Mapping Core" \
  "Empty-scan guards, filtered-empty guards, scan diagnostics, and no invalid LiDAR factors."

create_milestone \
  "M1: PointAttributes v1" \
  "PR290-style attribute container: intensity, timestamp, line, tag, scanner_id, rgb, static/dynamic labels."

create_milestone \
  "M2: Colored PCD Export" \
  "Export XYZ/XYZI/XYZRGB maps with intensity/scanner/height/time/static-dynamic coloring."

create_milestone \
  "M3: MapCleaner Dataset Export" \
  "Export individual scan frames, traj_lidar.txt, metadata.json for MapCleaner-compatible offline cleaning."

create_milestone \
  "M4: Diagnostics and QualitySnapshot" \
  "Unified scan guard, loop, wheel, GNSS, export, and degeneracy diagnostics."

create_issue_if_missing \
  "M0.1 ScanGuard: reject empty raw and filtered point clouds safely" \
  "M0: Safe Mapping Core" \
  "area:core,area:preprocess,type:milestone-task,priority:p0" \
  "Add a core ScanGuard utility and integrate it before LiDAR update/factor creation. Acceptance: empty raw scans and filtered-empty scans do not crash, do not create keyframes/factors, and emit structured diagnostics."

create_issue_if_missing \
  "M0.2 ScanGuard config: min_raw_points, min_filtered_points, burst handling" \
  "M0: Safe Mapping Core" \
  "area:core,area:preprocess,type:milestone-task" \
  "Add scan_guard config block and sane defaults. Acceptance: thresholds are visible in config and logs."

create_issue_if_missing \
  "M0.3 ScanGuard diagnostics topic/logs" \
  "M0: Safe Mapping Core" \
  "area:diagnostics,type:milestone-task" \
  "Publish or log ScanGuardStatus with raw_points, filtered_points, skip reason, and consecutive skipped frames."

create_issue_if_missing \
  "M1.1 PointAttributes: standard attribute schema" \
  "M1: PointAttributes v1" \
  "area:attributes,area:core,type:milestone-task" \
  "Add PR290-style standard point attribute schema: intensity, timestamp, line, tag, scanner_id, rgb, normals/covariances placeholders."

create_issue_if_missing \
  "M2.1 PCD exporter: XYZ, XYZI, XYZRGB" \
  "M2: Colored PCD Export" \
  "area:export,type:milestone-task" \
  "Add PCD export variants and color modes."

create_issue_if_missing \
  "M3.1 MapCleaner-compatible scan dataset writer" \
  "M3: MapCleaner Dataset Export" \
  "area:mapcleaner,area:export,type:milestone-task" \
  "Export scans/*.pcd, traj_lidar.txt, and metadata.json compatible with offline MapCleaner pipeline."

create_issue_if_missing \
  "M4.1 QualitySnapshot base message/log schema" \
  "M4: Diagnostics and QualitySnapshot" \
  "area:diagnostics,type:milestone-task" \
  "Define QualitySnapshot base schema and first publisher/log sink."
