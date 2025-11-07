terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.33"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.14"
    }
  }
}

provider "kubernetes" {
  config_path = "~/.kube/config"
}

provider "helm" {
  kubernetes {
    config_path = "~/.kube/config"
  }
}

# namespace already created manually / by helm:
# we just tell helm to use it, not create it
resource "helm_release" "aerocast" {
  name       = "aerocast"
  chart      = "../helm/aerocast"
  namespace  = "aerocast"
  create_namespace = false
  # don’t block forever in local/kind
  wait    = false
  timeout = 600
}
