# Generated by Django 5.1 on 2024-08-10 11:03

import notification.models.message
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("notification", "0014_alter_directimagemessage_image_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="directimagemessage",
            name="image",
            field=models.ImageField(
                blank=True,
                max_length=500,
                null=True,
                upload_to=notification.models.message.user_image_upload_path,
            ),
        ),
        migrations.AlterField(
            model_name="directvideomessage",
            name="video",
            field=models.FileField(
                blank=True,
                max_length=500,
                null=True,
                upload_to=notification.models.message.user_video_upload_path,
            ),
        ),
        migrations.AlterField(
            model_name="groupimagemessage",
            name="image",
            field=models.ImageField(
                blank=True,
                max_length=500,
                null=True,
                upload_to=notification.models.message.user_image_upload_path,
            ),
        ),
        migrations.AlterField(
            model_name="groupvideomessage",
            name="video",
            field=models.FileField(
                blank=True,
                max_length=500,
                null=True,
                upload_to=notification.models.message.user_video_upload_path,
            ),
        ),
    ]