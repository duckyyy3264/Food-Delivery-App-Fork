# Generated by Django 5.1 on 2024-09-23 12:29

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("food", "0011_dish_total_orders"),
    ]

    operations = [
        migrations.AddField(
            model_name="dish",
            name="is_disabled",
            field=models.BooleanField(blank=True, default=False, null=True),
        ),
    ]
