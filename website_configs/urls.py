"""website_configs URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from app_music_classification_api import views

urlpatterns = [
    path('api', views.my_api, name='api'),
    path('get_types', views.get_types, name='get_types'),
    path('types_img', views.types_img, name='types_img'),
    path('songlist_img', views.songlist_img, name='songlist_img'),
    path('add', views.new_train_data, name='new_train_data'),
    path('download', views.dataset_download, name='dataset_download'),
]