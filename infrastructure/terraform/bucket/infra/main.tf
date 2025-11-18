//
// Create a new IAM Service Account (SA).
//
resource "yandex_iam_service_account" "sa" {
  name        = var.sa_name
  description = "Service Account for OTUS Terraform course"
}

//
// Create a new IAM Service Account IAM Member.
//
resource "yandex_resourcemanager_folder_iam_member" "sa-role" {
  folder_id          = var.yc_folder_id
  role               = "storage.admin"
  member             = "serviceAccount:${yandex_iam_service_account.sa.id}"
}

//
// Create a new IAM Service Account Static Access SKey.
//
resource "yandex_iam_service_account_static_access_key" "sa-static-key" {
  service_account_id = yandex_iam_service_account.sa.id
  description        = "static access key for object storage"
}

//
// Create a new Storage Bucket. 
//
resource "yandex_storage_bucket" "bucket" {
  bucket = var.bucket_name
  access_key = yandex_iam_service_account_static_access_key.sa-static-key.access_key
  secret_key = yandex_iam_service_account_static_access_key.sa-static-key.secret_key
}
