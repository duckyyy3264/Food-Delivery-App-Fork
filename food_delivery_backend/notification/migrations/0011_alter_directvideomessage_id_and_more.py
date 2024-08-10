# Generated by Django 5.1 on 2024-08-09 06:36

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("notification", "0010_alter_directmessage_room_alter_groupmessage_room"),
    ]

    operations = [
        migrations.AlterField(
            model_name="directvideomessage",
            name="id",
            field=models.UUIDField(
                db_index=True,
                default=uuid.uuid4,
                editable=False,
                primary_key=True,
                serialize=False,
            ),
        ),
        migrations.AlterField(
            model_name="groupvideomessage",
            name="id",
            field=models.UUIDField(
                db_index=True,
                default=uuid.uuid4,
                editable=False,
                primary_key=True,
                serialize=False,
            ),
        ),
    ]