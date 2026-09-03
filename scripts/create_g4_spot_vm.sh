#!/usr/bin/env bash

set -euo pipefail

account="greg.friedland@gmail.com"
boot_disk_size_gb=200
dry_run=false
instance_name="bfgs-g4-spot"
machine_type="g4-standard-48"
project="muziq-501806"
region="us-east5"
zone=""

usage() {
    cat <<'EOF'
Usage: create_g4_spot_vm.sh [OPTIONS]

Create a GCP g4-standard-48 Spot VM.

Options:
  --project PROJECT       GCP project ID (default: muziq-501806)
  --region REGION         GCP region (default: us-east5)
  --zone ZONE             Exact zone; otherwise discover one in REGION
  --name NAME             Instance name (default: bfgs-g4-spot)
  --account ACCOUNT       gcloud account (default: greg.friedland@gmail.com)
  --boot-disk-size-gb GB  Boot disk size (default: 200)
  --dry-run               Print the create command without provisioning
  -h, --help              Show this help
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "$value" ]] || die "$option requires a value"
}

while (($# > 0)); do
    case "$1" in
        --project | --region | --zone | --name | --account | --boot-disk-size-gb)
            require_value "$1" "${2:-}"
            case "$1" in
                --project) project="$2" ;;
                --region) region="$2" ;;
                --zone) zone="$2" ;;
                --name) instance_name="$2" ;;
                --account) account="$2" ;;
                --boot-disk-size-gb) boot_disk_size_gb="$2" ;;
            esac
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ "$boot_disk_size_gb" =~ ^[0-9]+$ ]] || die "disk size must be an integer"
((boot_disk_size_gb >= 40)) || die "disk size must be at least 40 GB"
[[ -z "$zone" || "$zone" == "$region"-* ]] || \
    die "zone $zone is not in region $region"
command -v gcloud >/dev/null 2>&1 || die "gcloud is not installed"

credential="$({
    gcloud auth list \
        --filter="account=$account" \
        --format='value(account)'
} 2>/dev/null)"
[[ "$credential" == "$account" ]] || \
    die "authenticate first: gcloud auth login $account"

gcloud --account="$account" projects describe "$project" \
    --format='value(projectId)' >/dev/null

if [[ -z "$zone" ]]; then
    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] || continue
        if gcloud --account="$account" --project="$project" \
            compute machine-types describe "$machine_type" \
            --zone="$candidate" >/dev/null 2>&1; then
            zone="$candidate"
            break
        fi
    done < <(
        gcloud --account="$account" --project="$project" compute zones list \
            --filter="region.basename()=$region AND status=UP" \
            --format='value(name)' | LC_ALL=C sort
    )
fi

[[ -n "$zone" ]] || die "$machine_type is unavailable in region $region"
gcloud --account="$account" --project="$project" compute machine-types \
    describe "$machine_type" --zone="$zone" >/dev/null

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
startup_script="$script_dir/g4_startup.sh"
[[ -r "$startup_script" ]] || die "missing startup script: $startup_script"

create_command=(
    gcloud
    --account="$account"
    --project="$project"
    compute instances create "$instance_name"
    --zone="$zone"
    --machine-type="$machine_type"
    --image-family=ubuntu-2404-lts-amd64
    --image-project=ubuntu-os-cloud
    --boot-disk-type=hyperdisk-balanced
    --boot-disk-size="${boot_disk_size_gb}GB"
    --boot-disk-provisioned-iops=3000
    --boot-disk-provisioned-throughput=140
    --provisioning-model=SPOT
    --instance-termination-action=DELETE
    --maintenance-policy=TERMINATE
    --no-restart-on-failure
    --no-service-account
    --no-scopes
    --no-shielded-secure-boot
    --shielded-vtpm
    --shielded-integrity-monitoring
    --metadata-from-file="startup-script=$startup_script"
)

printf 'Project: %s\nRegion: %s\nZone: %s\nInstance: %s\n' \
    "$project" "$region" "$zone" "$instance_name"

if [[ "$dry_run" == true ]]; then
    printf '%q ' "${create_command[@]}"
    printf '\n'
    exit 0
fi

"${create_command[@]}"
gcloud --account="$account" --project="$project" compute instances describe \
    "$instance_name" --zone="$zone" \
    --format='table(name,zone.basename(),status,machineType.basename(),scheduling.provisioningModel)'
