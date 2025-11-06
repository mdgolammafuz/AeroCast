terraform {
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.33"
    }
    helm = {
      source = "hashicorp/helm"
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

resource "kubernetes_namespace" "aerocast" {
  metadata {
    name = "aerocast"
  }
}

resource "helm_release" "aerocast" {
  name       = "aerocast"
  repository = "file://../helm/aerocast"
  chart      = "aerocast"
  namespace  = kubernetes_namespace.aerocast.metadata[0].name
}
