# Generated by Django 5.1.2 on 2024-10-16 16:39

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Locations",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(max_length=500)),
                ("category", models.CharField(blank=True, max_length=500, null=True)),
                (
                    "description",
                    models.CharField(blank=True, max_length=300, null=True),
                ),
                ("OpeningTime", models.JSONField(blank=True, null=True)),
                ("website_address", models.URLField(blank=True, null=True)),
                (
                    "phone_number",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                ("zipcode", models.CharField(blank=True, max_length=200, null=True)),
                ("city", models.CharField(blank=True, max_length=200, null=True)),
                ("country", models.CharField(blank=True, max_length=200, null=True)),
                ("address", models.CharField(blank=True, max_length=200, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True, null=True)),
                ("edited_at", models.DateTimeField(auto_now=True)),
                ("toilet", models.BooleanField(null=True)),
                ("accessibility", models.BooleanField(null=True)),
                ("animals", models.BooleanField(null=True)),
                (
                    "image",
                    models.ImageField(
                        blank=True, null=True, upload_to="static/images/"
                    ),
                ),
                ("lat", models.CharField(blank=True, max_length=200, null=True)),
                ("lng", models.CharField(blank=True, max_length=200, null=True)),
                ("place_id", models.CharField(blank=True, max_length=200, null=True)),
                ("icon_image", models.CharField(blank=True, max_length=200, null=True)),
                ("average_rating", models.FloatField(blank=True, null=True)),
                ("user_rating_accessibility", models.FloatField(blank=True, null=True)),
                ("num_positive_comments", models.IntegerField(default=0)),
                ("num_negative_comments", models.IntegerField(default=0)),
                (
                    "wheelchair_accessible_entrance",
                    models.BooleanField(blank=True, null=True),
                ),
                ("visitTime", models.CharField(blank=True, max_length=200, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Place",
            fields=[
                ("ID", models.IntegerField(primary_key=True, serialize=False)),
                ("Name", models.CharField(max_length=255)),
                ("Category", models.CharField(max_length=100)),
                ("Description", models.TextField()),
                ("OpeningTime", models.CharField(max_length=500)),
                ("Website", models.URLField(blank=True, null=True)),
                ("PhoneNumber", models.CharField(blank=True, max_length=20, null=True)),
                ("Location", models.CharField(max_length=100)),
                ("Address", models.CharField(max_length=255)),
                ("Toilet", models.BooleanField(default=False)),
                ("Accessibility", models.BooleanField(default=False)),
                ("Animals", models.BooleanField(default=False)),
                ("Image", models.CharField(max_length=250)),
                ("Icon", models.CharField(max_length=250)),
                ("VisitTime", models.CharField(max_length=100)),
                ("Map_Priority", models.IntegerField(default=0)),
                ("place_id", models.CharField(blank=True, max_length=200, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="TravelPreference",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("departure_date", models.DateField()),
                ("departure_time", models.CharField(max_length=5)),
                (
                    "moving_preference",
                    models.CharField(
                        choices=[
                            ("car", "Car"),
                            ("bus", "Bus"),
                            ("train", "Train"),
                            ("plane", "Plane"),
                            ("bike", "Bike"),
                            ("walking", "Walking"),
                            ("other", "Other"),
                        ],
                        max_length=20,
                    ),
                ),
                (
                    "main_interests",
                    models.CharField(
                        choices=[
                            ("nature", "Nature and Outdoor Activities"),
                            ("culture", "Cultural Experiences"),
                            ("adventure", "Adventure Sports"),
                            ("food", "Food and Culinary"),
                            ("history", "Historical Sites"),
                            ("shopping", "Shopping"),
                            ("other", "Other"),
                        ],
                        max_length=50,
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Distances",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("mode", models.CharField(blank=True, max_length=200, null=True)),
                ("distance_km", models.DecimalField(decimal_places=2, max_digits=10)),
                ("duration_mins", models.DecimalField(decimal_places=2, max_digits=10)),
                (
                    "duration_traffic_mins",
                    models.DecimalField(
                        blank=True, decimal_places=2, max_digits=10, null=True
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True, null=True)),
                ("edited_at", models.DateTimeField(auto_now=True)),
                (
                    "from_location",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="from_location",
                        to="cagliaritour.locations",
                    ),
                ),
                (
                    "to_location",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="to_location",
                        to="cagliaritour.locations",
                    ),
                ),
            ],
        ),
    ]
