#!/usr/bin/env bash

set -euo pipefail

account=""
boot_disk_size_gb=50
dry_run=false
instance_name="bfgs-g4-spot"
machine_type="g4-standard-48"
project=""
region="us-east5"
zone=""

usage() {
    cat <<'EOF'
Usage: manage_g4_spot_vm.sh ACTION --project PROJECT --account ACCOUNT [OPTIONS]

Create, stop, or start a GCP g4-standard-48 Spot VM.

Actions:
  create                  Create and start the VM
  stop                    Stop the VM while preserving its boot disk
  start                   Start a stopped VM

Options:
  --project PROJECT       GCP project ID (required)
  --account ACCOUNT       Authenticated gcloud account (required)
  --region REGION         GCP region (default: us-east5)
  --zone ZONE             Exact zone; otherwise discover it
  --name NAME             Instance name (default: bfgs-g4-spot)
  --boot-disk-size-gb GB  Boot disk size for create (default: 50)
  --dry-run               Print the mutating command without running it
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

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

if (($# == 0)); then
    usage >&2
    exit 1
fi

action="$1"
shift
case "$action" in
    create | stop | start) ;;
    -h | --help)
        usage
        exit 0
        ;;
    *) die "unknown action: $action" ;;
esac

while (($# > 0)); do
    case "$1" in
        --project | --account | --region | --zone | --name | --boot-disk-size-gb)
            require_value "$1" "${2:-}"
            case "$1" in
                --project) project="$2" ;;
                --account) account="$2" ;;
                --region) region="$2" ;;
                --zone) zone="$2" ;;
                --name) instance_name="$2" ;;
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

[[ -n "$project" ]] || die "--project is required"
[[ -n "$account" ]] || die "--account is required"
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

if [[ "$action" == create && -z "$zone" ]]; then
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

if [[ "$action" != create && -z "$zone" ]]; then
    zones="$({
        gcloud --account="$account" --project="$project" \
            compute instances list \
            --filter="name=$instance_name AND zone:$region-" \
            --format='value(zone.basename())'
    })"
    [[ -n "$zones" ]] || die "instance not found in region $region"
    [[ "$zones" != *$'\n'* ]] || die "instance name exists in multiple zones"
    zone="$zones"
fi

[[ -n "$zone" ]] || die "$machine_type is unavailable in region $region"

if [[ "$action" == create ]]; then
    gcloud --account="$account" --project="$project" compute machine-types \
        describe "$machine_type" --zone="$zone" >/dev/null
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    startup_script="$script_dir/g4_startup.sh"
    [[ -r "$startup_script" ]] || die "missing startup script: $startup_script"
    command=(
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
        --no-boot-disk-auto-delete
        --provisioning-model=SPOT
        --instance-termination-action=STOP
        --maintenance-policy=TERMINATE
        --no-restart-on-failure
        --no-service-account
        --no-scopes
        --no-shielded-secure-boot
        --shielded-vtpm
        --shielded-integrity-monitoring
        --metadata-from-file="startup-script=$startup_script"
    )
else
    status="$({
        gcloud --account="$account" --project="$project" \
            compute instances describe "$instance_name" --zone="$zone" \
            --format='value(status)'
    })"
    if [[ "$action" == stop ]]; then
        if [[ "$status" == TERMINATED ]]; then
            printf '%s is already stopped in %s\n' "$instance_name" "$zone"
            exit 0
        fi
        [[ "$status" == RUNNING ]] || die "cannot stop VM in state $status"
        command=(
            gcloud --account="$account" --project="$project"
            compute instances stop "$instance_name" --zone="$zone"
        )
    else
        if [[ "$status" == RUNNING ]]; then
            printf '%s is already running in %s\n' "$instance_name" "$zone"
            exit 0
        fi
        [[ "$status" == TERMINATED ]] || die "cannot start VM in state $status"
        command=(
            gcloud --account="$account" --project="$project"
            compute instances start "$instance_name" --zone="$zone"
        )
    fi
fi

printf 'Action: %s\nProject: %s\nRegion: %s\nZone: %s\nInstance: %s\n' \
    "$action" "$project" "$region" "$zone" "$instance_name"
if [[ "$dry_run" == true ]]; then
    print_command "${command[@]}"
    exit 0
fi

"${command[@]}"
gcloud --account="$account" --project="$project" compute instances describe \
    "$instance_name" --zone="$zone" \
    --format='table(name,zone.basename(),status,machineType.basename(),scheduling.provisioningModel)'
