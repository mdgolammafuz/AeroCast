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

resource "helm_release" "aerocast" {
  name             = "aerocast"
  chart            = "../helm/aerocast"
  namespace        = "aerocast"
  create_namespace = true
  
  # Don't block forever in local/kind if pods take time to pull
  wait    = false
  timeout = 600
}