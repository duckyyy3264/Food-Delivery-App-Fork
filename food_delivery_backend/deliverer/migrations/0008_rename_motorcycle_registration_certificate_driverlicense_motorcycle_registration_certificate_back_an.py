# Generated by Django 5.1 on 2024-09-17 12:14

import deliverer.models.driver_license
import deliverer.models.other_info
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("deliverer", "0007_remove_driverlicense_license_back_and_more"),
    ]

    operations = [
        migrations.RenameField(
            model_name="driverlicense",
            old_name="motorcycle_registration_certificate",
            new_name="motorcycle_registration_certificate_back",
        ),
        migrations.AddField(
            model_name="driverlicense",
            name="motorcycle_registration_certificate_front",
            field=models.ImageField(
                blank=True,
                null=True,
                upload_to=deliverer.models.driver_license.registration_certificate_image_upload_path,
            ),
        ),
        migrations.AlterField(
            model_name="otherinfo",
            name="judicial_record",
            field=models.ImageField(
                blank=True, null=True, upload_to=deliverer.models.other_info.other_info_image_upload_path
            ),
        ),
    ]
