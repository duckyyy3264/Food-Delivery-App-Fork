# Generated by Django 5.1 on 2024-09-03 01:55

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("order", "0036_alter_delivery_order"),
    ]

    operations = [
        migrations.AlterModelOptions(
            name="delivery",
            options={"ordering": ["-created_at"]},
        ),
    ]